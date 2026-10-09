#include "storage/ducklake_sorted_merge.hpp"

#include "duckdb/common/bswap.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/scan_read_ahead.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "functions/ducklake_compaction_functions.hpp"
#include "storage/ducklake_multi_file_list.hpp"
#include "storage/ducklake_table_entry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Single File Scan
//===--------------------------------------------------------------------===//
DuckLakeFileScanBind::DuckLakeFileScanBind(ClientContext &context, DuckLakeTableEntry &table,
                                           const DuckLakeFileListEntry &file_p)
    : file(file_p) {
	EntryLookupInfo info(CatalogType::TABLE_ENTRY, QualifiedName(table.name));
	function = table.GetScanFunction(context, bind_data, info);
	auto &read_info = function.function_info->Cast<DuckLakeFunctionInfo>();
	vector<DuckLakeFileListEntry> files {file};
	bind_data->Cast<MultiFileBindData>().file_list = make_uniq<DuckLakeMultiFileList>(read_info, std::move(files));
}

//! Row groups each single-file scan reads ahead
static constexpr idx_t SCANNER_READ_AHEAD_DEPTH = 2;

DuckLakeSingleFileScanner::DuckLakeSingleFileScanner(ClientContext &context, DuckLakeTableEntry &table,
                                                     const DuckLakeFileScanBind &scan_bind, vector<column_t> column_ids,
                                                     unique_ptr<TableFilterSet> filters_p)
    : context(context), function(scan_bind.function), bind_data(scan_bind.bind_data->Copy()),
      filters(std::move(filters_p)) {
	auto virtual_columns = table.GetVirtualColumns();
	for (auto &column_id : column_ids) {
		if (IsVirtualColumn(column_id)) {
			types.push_back(virtual_columns.at(column_id).type);
		} else {
			types.push_back(table.GetColumns().GetColumn(PhysicalIndex(column_id)).Type());
		}
	}
	thread_context = make_uniq<ThreadContext>(context);
	execution_context = make_uniq<ExecutionContext>(context, *thread_context, nullptr);
	TableFunctionInitInput input(bind_data.get(), std::move(column_ids), vector<idx_t>(), filters.get());
	global_state = function.init_global(context, input);
	// DuckDB's automatic read-ahead reserves memory for every thread of each scan, and a merge runs many
	// single-threaded scans at once, so read ahead a fixed depth instead
	auto &read_ahead = global_state->Cast<MultiFileGlobalState>().read_ahead;
	if (read_ahead && Settings::Get<ReadAheadDepthSetting>(context) < 0) {
		read_ahead = make_uniq<ScanReadAhead>(context, SCANNER_READ_AHEAD_DEPTH, nullptr);
	}
	local_state = function.init_local(*execution_context, input, global_state.get());
}

bool DuckLakeSingleFileScanner::Scan(DataChunk &chunk) {
	while (true) {
		chunk.Reset();
		TableFunctionInput input(bind_data.get(), local_state.get(), global_state.get());
		input.async_result = AsyncResultType::IMPLICIT;
		function.function(context, input, chunk);
		if (chunk.size() > 0) {
			return true;
		}
		auto result_type = input.async_result.GetResultType();
		if (result_type == AsyncResultType::BLOCKED) {
			input.async_result.ExecuteTasksSynchronously();
		} else if (result_type != AsyncResultType::HAVE_MORE_OUTPUT) {
			return false;
		}
	}
}

//===--------------------------------------------------------------------===//
// Worker Policy
//===--------------------------------------------------------------------===//
//! Batches a worker keeps in the batch copy, and the fraction of the memory limit the batch copy takes
static constexpr idx_t BATCHES_PER_WORKER = 2;
static constexpr idx_t WRITER_MEMORY_FRACTION = 4;
//! A narrower batch moves the average 1/BATCH_BYTES_DECAY of the way
static constexpr idx_t BATCH_BYTES_DECAY = 12;

DuckLakeSortedMergeWorkerPolicy::DuckLakeSortedMergeWorkerPolicy(idx_t max_threads)
    : max_threads(MaxValue<idx_t>(max_threads, 1)) {
}

void DuckLakeSortedMergeWorkerPolicy::AddBatch(idx_t batch_bytes, idx_t max_memory) {
	if (batch_bytes >= average_batch_bytes) {
		average_batch_bytes = batch_bytes;
	} else {
		average_batch_bytes = (average_batch_bytes * (BATCH_BYTES_DECAY - 1) + batch_bytes) / BATCH_BYTES_DECAY;
	}
	auto target = TargetWorkers(max_memory);
	limit = target < limit ? target : MinValue(target, limit * 2);
}

idx_t DuckLakeSortedMergeWorkerPolicy::WorkerLimit(idx_t max_memory) const {
	if (average_batch_bytes == 0) {
		// nothing measured yet - one worker measures the first batch
		return 1;
	}
	// the memory limit may have shrunk since the last batch
	return MinValue(limit, TargetWorkers(max_memory));
}

idx_t DuckLakeSortedMergeWorkerPolicy::TargetWorkers(idx_t max_memory) const {
	auto workers = max_memory / (BATCHES_PER_WORKER * WRITER_MEMORY_FRACTION * MaxValue<idx_t>(average_batch_bytes, 1));
	return MaxValue<idx_t>(1, MinValue(workers, max_threads));
}

//===--------------------------------------------------------------------===//
// Merge Sequencer
//===--------------------------------------------------------------------===//
//! Read up to 8 bytes as a big-endian integer, zero-padded on the right
static uint64_t LoadKeyWord(const_data_ptr_t data, idx_t size) {
	if (size >= sizeof(uint64_t)) {
		return BSwapIfLE(Load<uint64_t>(data));
	}
	uint64_t result = 0;
	for (idx_t i = 0; i < size; i++) {
		result |= uint64_t(data[i]) << (8 * (sizeof(uint64_t) - 1 - i));
	}
	return result;
}

//! A sort key with its first 16 bytes as big-endian integers, so most comparisons are two integer compares
struct MergeKey {
	MergeKey() = default;
	explicit MergeKey(string_t key_p) : key(key_p) {
		auto data = const_data_ptr_cast(key.GetData());
		auto size = key.GetSize();
		high = LoadKeyWord(data, size);
		if (size > sizeof(uint64_t)) {
			low = LoadKeyWord(data + sizeof(uint64_t), size - sizeof(uint64_t));
		}
	}

	uint64_t high = 0;
	uint64_t low = 0;
	string_t key;

	bool operator<(const MergeKey &other) const {
		if (high != other.high) {
			return high < other.high;
		}
		if (low != other.low) {
			return low < other.low;
		}
		if (PrefixIsKey(other)) {
			return false;
		}
		return GreaterThan::Operation(other.key, key);
	}

	bool operator==(const MergeKey &other) const {
		if (high != other.high || low != other.low) {
			return false;
		}
		return PrefixIsKey(other) || Equals::Operation(key, other.key);
	}

private:
	//! Whether equal prefixes mean equal keys
	bool PrefixIsKey(const MergeKey &other) const {
		return key.GetSize() == other.key.GetSize() && key.GetSize() <= 16;
	}
};

//! The sort keys of a chunk of an input, in file order
struct DuckLakeSortedMergeSequencer::KeyChunk {
	shared_ptr<Vector> sort_keys;
	vector<MergeKey> keys;
	vector<idx_t> row_numbers;
};

//! The keys of file rows [row_start, row_end) of an input, loaded a chunk at a time by whichever thread gets to it
struct DuckLakeSortedMergeSequencer::KeySegment {
	KeySegment(idx_t row_start, idx_t row_end) : row_start(row_start), row_end(row_end) {
	}

	idx_t row_start;
	idx_t row_end;
	mutex lock;
	unique_ptr<DuckLakeSingleFileScanner> scanner;
	DataChunk scan_chunk;
	DataChunk key_chunk;
	unique_ptr<ExpressionExecutor> executor;
	deque<unique_ptr<KeyChunk>> chunks;
	//! Readable without the lock, to find segments worth loading
	atomic<idx_t> loaded_chunks {0};
	atomic<bool> done {false};
};

struct DuckLakeSortedMergeSequencer::MergeInput {
	idx_t input_idx;
	//! Guards segments and next_segment
	mutex segments_lock;
	//! The segment being merged, then the segments loaded ahead
	deque<shared_ptr<KeySegment>> segments;
	idx_t next_segment = 0;
	idx_t segment_count = 0;
	//! Rows per segment, all rows in one segment when the file's row count is not known
	idx_t segment_rows = 0;

	//! The rest is only used by the merge
	shared_ptr<Vector> sort_keys;
	vector<MergeKey> keys;
	vector<idx_t> row_numbers;
	idx_t position = 0;
	//! The last key of the previous chunk, to verify the input is sorted across chunks
	string previous_key_data;
	MergeKey previous_key;
	bool has_previous_key = false;
	bool exhausted = false;

	const MergeKey &CurrentKey() const {
		return keys[position];
	}

	//! Create the next segment, the caller holds segments_lock
	shared_ptr<KeySegment> AddSegment() {
		auto row_start = next_segment * segment_rows;
		auto row_end = next_segment + 1 == segment_count ? NumericLimits<idx_t>::Maximum() : row_start + segment_rows;
		next_segment++;
		auto segment = make_shared_ptr<KeySegment>(row_start, row_end);
		segments.push_back(segment);
		return segment;
	}
};

//! Chunks a segment loads ahead of the merge
static constexpr idx_t CHUNKS_AHEAD = 4;
//! Segments an input loads ahead of the merge
static constexpr idx_t MAX_INPUT_SEGMENTS_AHEAD = 2;

DuckLakeSortedMergeSequencer::DuckLakeSortedMergeSequencer(ClientContext &context, DuckLakeTableEntry &table,
                                                           const vector<unique_ptr<DuckLakeFileScanBind>> &inputs,
                                                           idx_t segment_size)
    : context(context), table(table), inputs(inputs), segment_size(MaxValue<idx_t>(segment_size, 1)) {
	auto sort_data = table.GetSortData();
	if (!sort_data) {
		throw InvalidInputException("Table %s has no sort key to merge on", table.name);
	}
	auto binder = Binder::CreateBinder(context);
	TableIndex table_index(0);
	auto orders = DuckLakeCompactor::BindSortOrders(*binder, table.GetColumns(), table.name, table_index,
	                                                DuckLakeCompactor::ParseSortOrders(*sort_data));
	if (orders.empty()) {
		throw InvalidInputException("Table %s has no sort key in the DuckDB dialect to merge on", table.name);
	}

	// scan only the columns the sort key reads
	set<idx_t> key_columns;
	for (auto &order : orders) {
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
		    *order.expression,
		    [&](const BoundColumnRefExpression &colref) { key_columns.insert(colref.Binding().column_index); });
	}
	vector<ColumnBinding> bindings;
	vector<LogicalType> binding_types;
	for (auto &column_idx : key_columns) {
		scan_column_ids.push_back(column_idx);
		bindings.emplace_back(table_index, ProjectionIndex(column_idx));
		binding_types.push_back(table.GetColumns().GetColumn(PhysicalIndex(column_idx)).Type());
	}
	scan_column_ids.push_back(MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER);

	ColumnBindingResolver resolver(std::move(bindings), std::move(binding_types));
	set<idx_t> decodable;
	for (auto &order : orders) {
		resolver.VisitExpression(&order.expression);
		if (order.expression->GetExpressionType() == ExpressionType::BOUND_REF) {
			auto column_idx = scan_column_ids[order.expression->Cast<BoundReferenceExpression>().Index()];
			if (decodable.insert(column_idx).second) {
				decodable_columns.push_back(column_idx);
			}
		}
		modifiers.emplace_back(order.type, order.null_order);
		key_types.push_back(order.expression->GetReturnType());
		key_expressions.push_back(std::move(order.expression));
	}
	if (decodable_columns.size() != key_expressions.size()) {
		decodable_columns.clear();
	}

	for (idx_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
		auto input = make_uniq<MergeInput>();
		input->input_idx = input_idx;
		auto &row_count = inputs[input_idx]->row_count;
		if (row_count.IsValid()) {
			input->segment_rows = this->segment_size;
			input->segment_count = (row_count.GetIndex() + this->segment_size - 1) / this->segment_size;
		} else {
			input->segment_rows = NumericLimits<idx_t>::Maximum();
			input->segment_count = 1;
		}
		merge_inputs.push_back(std::move(input));
	}
}

DuckLakeSortedMergeSequencer::~DuckLakeSortedMergeSequencer() {
}

bool DuckLakeSortedMergeSequencer::HasMergeableSort(DuckLakeTableEntry &table) {
	auto sort_data = table.GetSortData();
	return sort_data && !DuckLakeCompactor::ParseSortOrders(*sort_data).empty();
}

void DuckLakeSortedMergeSequencer::Initialize() {
	for (auto &input : merge_inputs) {
		if (LoadChunk(*input)) {
			PushInput(input->input_idx);
		}
	}
	initialized = true;
}

void DuckLakeSortedMergeSequencer::LoadKeys(MergeInput &input, KeySegment &segment) {
	if (!segment.scanner) {
		unique_ptr<TableFilterSet> filters;
		if (input.segment_count > 1) {
			filters = make_uniq<TableFilterSet>();
			ProjectionIndex row_number_column(scan_column_ids.size() - 1);
			filters->PushFilter(row_number_column, ExpressionFilter::CreateComparisonFilter(
			                                           ExpressionType::COMPARE_GREATERTHANOREQUALTO,
			                                           Value::BIGINT(NumericCast<int64_t>(segment.row_start))));
			if (segment.row_end != NumericLimits<idx_t>::Maximum()) {
				filters->PushFilter(row_number_column, ExpressionFilter::CreateComparisonFilter(
				                                           ExpressionType::COMPARE_LESSTHAN,
				                                           Value::BIGINT(NumericCast<int64_t>(segment.row_end))));
			}
		}
		segment.scanner = make_uniq<DuckLakeSingleFileScanner>(context, table, *inputs[input.input_idx],
		                                                       scan_column_ids, std::move(filters));
		segment.scan_chunk.Initialize(context, segment.scanner->GetTypes());
		segment.key_chunk.Initialize(context, key_types);
		segment.executor = make_uniq<ExpressionExecutor>(context, key_expressions);
	}
	if (!segment.scanner->Scan(segment.scan_chunk)) {
		segment.scanner.reset();
		segment.done = true;
		return;
	}
	auto count = segment.scan_chunk.size();
	auto chunk = make_uniq<KeyChunk>();
	segment.key_chunk.Reset();
	segment.executor->Execute(segment.scan_chunk, segment.key_chunk);
	chunk->sort_keys = make_shared_ptr<Vector>(LogicalType::BLOB, count);
	CreateSortKeyHelpers::CreateSortKey(segment.key_chunk, modifiers, *chunk->sort_keys);
	chunk->keys.reserve(count);
	for (auto entry : chunk->sort_keys->Values<string_t>()) {
		chunk->keys.emplace_back(entry.GetValue());
	}
	auto &row_number_vector = segment.scan_chunk.data[segment.scan_chunk.ColumnCount() - 1];
	chunk->row_numbers.reserve(count);
	for (auto entry : row_number_vector.Values<int64_t>()) {
		chunk->row_numbers.push_back(NumericCast<idx_t>(entry.GetValue()));
	}
	if (chunk->row_numbers.back() + 1 == segment.row_end) {
		// the last row of the segment is loaded, release the scanner without another scan
		segment.scanner.reset();
		segment.done = true;
	}
	segment.chunks.push_back(std::move(chunk));
	segment.loaded_chunks++;
}

bool DuckLakeSortedMergeSequencer::LoadChunk(MergeInput &input) {
	if (!input.keys.empty()) {
		input.previous_key_data = input.keys.back().key.GetString();
		input.previous_key = MergeKey(
		    string_t(input.previous_key_data.c_str(), UnsafeNumericCast<uint32_t>(input.previous_key_data.size())));
		input.has_previous_key = true;
	}
	input.keys.clear();
	input.row_numbers.clear();
	input.position = 0;
	unique_ptr<KeyChunk> chunk;
	while (!chunk) {
		shared_ptr<KeySegment> segment;
		{
			lock_guard<mutex> guard(input.segments_lock);
			if (input.segments.empty()) {
				if (input.next_segment == input.segment_count) {
					input.exhausted = true;
					return false;
				}
				input.AddSegment();
				segments_ahead++;
			}
			segment = input.segments.front();
		}
		lock_guard<mutex> segment_guard(segment->lock);
		if (segment->chunks.empty() && !segment->done) {
			LoadKeys(input, *segment);
		}
		if (!segment->chunks.empty()) {
			chunk = std::move(segment->chunks.front());
			segment->chunks.pop_front();
			segment->loaded_chunks--;
		} else {
			// every row of the segment is merged, or deleted
			lock_guard<mutex> guard(input.segments_lock);
			input.segments.pop_front();
			segments_ahead--;
		}
	}
	input.sort_keys = std::move(chunk->sort_keys);
	input.keys = std::move(chunk->keys);
	input.row_numbers = std::move(chunk->row_numbers);

	auto known_sorted = inputs[input.input_idx]->known_sorted;
#ifndef DEBUG
	// trust the catalog, debug builds verify it
	if (known_sorted) {
		return true;
	}
#endif
	auto count = input.keys.size();
	bool out_of_order = input.has_previous_key && input.keys[0] < input.previous_key;
	for (idx_t i = 1; i < count && !out_of_order; i++) {
		out_of_order = input.keys[i] < input.keys[i - 1];
	}
	if (out_of_order) {
		auto &path = inputs[input.input_idx]->file.file.path;
		if (known_sorted) {
			throw InvalidInputException("File \"%s\" is recorded as sorted on the sort key of table %s but is not",
			                            path, table.name);
		}
		throw InvalidInputException("Cannot merge file \"%s\": it is not sorted on the sort key of table %s", path,
		                            table.name);
	}
	return true;
}

bool DuckLakeSortedMergeSequencer::TopUpSegment(MergeInput &input) {
	vector<shared_ptr<KeySegment>> candidates;
	{
		lock_guard<mutex> guard(input.segments_lock);
		for (auto &segment : input.segments) {
			if (!segment->done && segment->loaded_chunks < CHUNKS_AHEAD) {
				candidates.push_back(segment);
			}
		}
	}
	for (auto &segment : candidates) {
		unique_lock<mutex> segment_guard(segment->lock, std::try_to_lock);
		if (segment_guard.owns_lock() && !segment->done && segment->loaded_chunks < CHUNKS_AHEAD) {
			LoadKeys(input, *segment);
			return true;
		}
	}
	return false;
}

bool DuckLakeSortedMergeSequencer::StartSegment() {
	optional_ptr<MergeInput> target;
	shared_ptr<KeySegment> segment;
	idx_t fewest_segments = MAX_INPUT_SEGMENTS_AHEAD + 1;
	for (auto &input : merge_inputs) {
		lock_guard<mutex> guard(input->segments_lock);
		if (input->next_segment < input->segment_count && input->segments.size() < fewest_segments) {
			fewest_segments = input->segments.size();
			target = input.get();
		}
	}
	if (!target) {
		return false;
	}
	{
		lock_guard<mutex> guard(target->segments_lock);
		if (target->next_segment == target->segment_count || segments_ahead >= max_segments_ahead) {
			return false;
		}
		segment = target->AddSegment();
		segments_ahead++;
	}
	unique_lock<mutex> segment_guard(segment->lock, std::try_to_lock);
	if (segment_guard.owns_lock() && !segment->done && segment->loaded_chunks == 0) {
		LoadKeys(*target, *segment);
	}
	return true;
}

bool DuckLakeSortedMergeSequencer::Prefetch() {
	auto input_count = merge_inputs.size();
	auto start = prefetch_cursor++;
	for (idx_t i = 0; i < input_count; i++) {
		if (TopUpSegment(*merge_inputs[(start + i) % input_count])) {
			return true;
		}
	}
	return segments_ahead < max_segments_ahead && StartSegment();
}

idx_t DuckLakeSortedMergeSequencer::RunEnd(const MergeInput &input, idx_t start, idx_t end,
                                           optional_ptr<MergeInput> next) const {
	if (!next) {
		return end;
	}
	auto &bound = next->CurrentKey();
	auto inclusive = input.input_idx < next->input_idx;
	auto before = [&](idx_t row) {
		auto &key = input.keys[row];
		return key < bound || (inclusive && key == bound);
	};
	// gallop to bracket the first row that does not come before the next input, then binary search
	idx_t lo = start;
	idx_t hi = end;
	idx_t step = 1;
	while (lo < end) {
		auto probe = MinValue(lo + step - 1, end - 1);
		if (!before(probe)) {
			hi = probe;
			break;
		}
		lo = probe + 1;
		step *= 2;
	}
	while (lo < hi) {
		auto mid = lo + (hi - lo) / 2;
		if (before(mid)) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	return lo;
}

bool DuckLakeSortedMergeSequencer::InputBefore(const MergeInput &lhs, const MergeInput &rhs) const {
	auto &lhs_key = lhs.CurrentKey();
	auto &rhs_key = rhs.CurrentKey();
	if (lhs_key < rhs_key) {
		return true;
	}
	return lhs.input_idx < rhs.input_idx && lhs_key == rhs_key;
}

void DuckLakeSortedMergeSequencer::PushInput(idx_t input_idx) {
	heap.push_back(input_idx);
	std::push_heap(heap.begin(), heap.end(),
	               [&](idx_t lhs, idx_t rhs) { return InputBefore(*merge_inputs[rhs], *merge_inputs[lhs]); });
}

void DuckLakeSortedMergeSequencer::SiftDownTop() {
	idx_t node = 0;
	while (true) {
		auto smallest = node;
		for (auto child = 2 * node + 1; child <= 2 * node + 2 && child < heap.size(); child++) {
			if (InputBefore(*merge_inputs[heap[child]], *merge_inputs[heap[smallest]])) {
				smallest = child;
			}
		}
		if (smallest == node) {
			return;
		}
		std::swap(heap[node], heap[smallest]);
		node = smallest;
	}
}

optional_idx DuckLakeSortedMergeSequencer::NextRow(idx_t input_idx) const {
	auto &input = *merge_inputs[input_idx];
	if (input.exhausted) {
		return optional_idx();
	}
	return input.row_numbers[input.position];
}

bool DuckLakeSortedMergeSequencer::NextBatch(idx_t max_rows, vector<DuckLakeMergeSlice> &slices,
                                             optional_ptr<DuckLakeMergeKeys> keys) {
	if (!initialized) {
		Initialize();
	}
	if (keys) {
		key_owners.assign(merge_inputs.size(), nullptr);
	}
	slices.clear();
	idx_t sequenced = 0;
	while (sequenced < max_rows && !heap.empty()) {
		auto &input = *merge_inputs[heap[0]];
		// the input to switch to once this one's current row no longer comes first
		optional_ptr<MergeInput> next_input;
		if (heap.size() > 1) {
			auto next_idx = heap[1];
			if (heap.size() > 2 && InputBefore(*merge_inputs[heap[2]], *merge_inputs[heap[1]])) {
				next_idx = heap[2];
			}
			next_input = merge_inputs[next_idx].get();
		}
		DuckLakeMergeSlice slice {input.input_idx, input.row_numbers[input.position], 0, 0};
		// the top input's current row is known to come first
		auto known_rows = idx_t(1);
		while (true) {
			auto end = MinValue(input.keys.size(), input.position + max_rows - sequenced);
			auto run_end = RunEnd(input, MinValue(input.position + known_rows, end), end, next_input);
			known_rows = 0;
			if (run_end > input.position) {
				if (keys) {
					if (key_owners[input.input_idx] != input.sort_keys.get()) {
						key_owners[input.input_idx] = input.sort_keys.get();
						keys->owners.push_back(input.sort_keys);
					}
					for (idx_t row = input.position; row < run_end; row++) {
						keys->keys.push_back(input.keys[row].key);
					}
				}
				slice.row_end = input.row_numbers[run_end - 1] + 1;
				slice.row_count += run_end - input.position;
				sequenced += run_end - input.position;
				input.position = run_end;
			}
			if (input.position < input.keys.size() || !LoadChunk(input) || sequenced == max_rows) {
				break;
			}
		}
		D_ASSERT(slice.row_count > 0);
		slices.push_back(slice);
		if (input.exhausted) {
			heap[0] = heap.back();
			heap.pop_back();
		}
		SiftDownTop();
	}
	return !slices.empty();
}

} // namespace duckdb
