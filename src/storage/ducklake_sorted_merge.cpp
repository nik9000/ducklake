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
#include "storage/ducklake_multisequence_select.hpp"
#include "storage/ducklake_table_entry.hpp"
#include "storage/ducklake_metadata_manager.hpp"
#include "storage/ducklake_stats.hpp"
#include "storage/ducklake_transaction.hpp"

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

//! The sort keys of a chunk of an input, in file order, or only its row numbers when its keys are not needed
struct DuckLakeSortedMergeKeyChunk {
	shared_ptr<Vector> sort_keys;
	vector<MergeKey> keys;
	vector<idx_t> row_numbers;
};
using KeyChunk = DuckLakeSortedMergeKeyChunk;

//! The keys of file rows [row_start, row_end) of an input, loaded a chunk at a time by whichever thread gets to it
struct DuckLakeSortedMergeSequencer::KeySegment {
	KeySegment(idx_t row_start, idx_t row_end, bool keyless = false)
	    : row_start(row_start), row_end(row_end), keyless(keyless) {
	}

	idx_t row_start;
	idx_t row_end;
	//! Whether only row numbers are read
	bool keyless;
	//! Held while scanning, guards the scanner
	mutex load_lock;
	unique_ptr<DuckLakeSingleFileScanner> scanner;
	DataChunk scan_chunk;
	DataChunk key_chunk;
	unique_ptr<ExpressionExecutor> executor;
	//! Guards chunks, so loaded chunks can be taken while the next one is scanned
	mutex chunks_lock;
	deque<shared_ptr<KeyChunk>> chunks;
	//! Readable without the lock, to find segments worth loading
	atomic<idx_t> loaded_chunks {0};
	//! Set after the last chunk is added to chunks
	atomic<bool> done {false};

	shared_ptr<KeyChunk> TakeLoaded() {
		lock_guard<mutex> guard(chunks_lock);
		if (chunks.empty()) {
			return nullptr;
		}
		auto chunk = std::move(chunks.front());
		chunks.pop_front();
		loaded_chunks--;
		return chunk;
	}
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

	//! The rest is only used by the merge: the chunks of keys taken from the segments, from the next row to sequence
	//! at row position of the first chunk on
	deque<shared_ptr<KeyChunk>> window;
	idx_t position = 0;
	//! The rows of the window from position on, and the end of each chunk of the window counted from its first row
	idx_t loaded = 0;
	vector<idx_t> window_ends;
	//! Whether every chunk of the input has been taken into the window
	bool fully_loaded = false;
	bool exhausted = false;
	//! The last key taken, to verify the input is sorted across chunks
	string previous_key_data;
	MergeKey previous_key;
	bool has_previous_key = false;
	//! The first and last key of the input, the last unknown when its last row is deleted or its row count is not known
	string first_key;
	string last_key;
	bool has_last_key = false;
	//! The section of inputs whose key ranges overlap that the input is merged in
	idx_t section = 0;
	//! Whether the input is merged alone and recorded as sorted, so its keys are not needed
	bool keyless = false;
	//! Whether the first chunk is taken, which waits for the input's section when the catalog bounds its keys
	bool started = false;

	//! The chunk and row in it of the row-th row of the window from position on
	pair<idx_t, idx_t> Locate(idx_t row) const {
		auto absolute = position + row;
		auto chunk_idx = NumericCast<idx_t>(std::upper_bound(window_ends.begin(), window_ends.end(), absolute) -
		                                    window_ends.begin());
		return make_pair(chunk_idx, chunk_idx == 0 ? absolute : absolute - window_ends[chunk_idx - 1]);
	}
	const MergeKey &KeyAt(idx_t row) const {
		auto location = Locate(row);
		return window[location.first]->keys[location.second];
	}
	idx_t RowNumberAt(idx_t row) const {
		auto location = Locate(row);
		return window[location.first]->row_numbers[location.second];
	}

	//! Create the next segment, the caller holds segments_lock
	shared_ptr<KeySegment> AddSegment() {
		auto row_start = next_segment * segment_rows;
		auto row_end = next_segment + 1 == segment_count ? NumericLimits<idx_t>::Maximum() : row_start + segment_rows;
		next_segment++;
		auto segment = make_shared_ptr<KeySegment>(row_start, row_end, keyless);
		segments.push_back(segment);
		return segment;
	}
};

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

static MergeKey KeyOf(const string &key_data) {
	return MergeKey(string_t(key_data.c_str(), UnsafeNumericCast<uint32_t>(key_data.size())));
}

void DuckLakeSortedMergeSequencer::ReadLastKey(MergeInput &input) {
	auto &row_count = inputs[input.input_idx]->row_count;
	if (!row_count.IsValid() || row_count.GetIndex() == 0) {
		return;
	}
	KeySegment last_row(row_count.GetIndex() - 1, NumericLimits<idx_t>::Maximum());
	LoadKeys(input, last_row);
	if (!last_row.chunks.empty()) {
		input.last_key = last_row.chunks.back()->keys.back().key.GetString();
		input.has_last_key = true;
	}
}

bool DuckLakeSortedMergeSequencer::CatalogBounds(const map<idx_t, DuckLakeFileKeyStats> &file_stats,
                                                 const vector<FieldIndex> &fields, MergeInput &input) {
	DataChunk bounds;
	bounds.Initialize(Allocator::DefaultAllocator(), key_types, 2);
	for (idx_t key_idx = 0; key_idx < fields.size(); key_idx++) {
		auto entry = file_stats.find(fields[key_idx].index);
		if (entry == file_stats.end()) {
			return false;
		}
		auto &stats = entry->second;
		auto &type = key_types[key_idx];
		auto ascending = modifiers[key_idx].order_type == OrderType::ASCENDING;
		auto nulls_first = modifiers[key_idx].null_type == OrderByNullType::NULLS_FIRST;
		// the column's values come first and last in the sort, NULLs before or after them
		auto lowest = ascending ? stats.min : stats.max;
		auto highest = ascending ? stats.max : stats.min;
		auto has_lowest = ascending ? stats.has_min : stats.has_max;
		auto has_highest = ascending ? stats.has_max : stats.has_min;
		auto lowest_is_exact = (ascending ? stats.min_is_exact : stats.max_is_exact) || RequiresValueComparison(type);
		auto highest_is_exact = (ascending ? stats.max_is_exact : stats.min_is_exact) || RequiresValueComparison(type);
		if (stats.contains_nan || has_lowest != has_highest) {
			return false;
		}
		Value low, high;
		if (has_lowest) {
			if (!lowest_is_exact || !highest_is_exact) {
				return false;
			}
			auto cast_low = Value(lowest).DefaultTryCastAs(type);
			auto cast_high = Value(highest).DefaultTryCastAs(type);
			if (!cast_low || !cast_high) {
				return false;
			}
			low = std::move(*cast_low);
			high = std::move(*cast_high);
		} else if (stats.null_count == 0) {
			return false;
		}
		// a column of only NULLs is NULL at both ends
		bounds.SetValue(key_idx, 0, stats.null_count > 0 && (nulls_first || !has_lowest) ? Value(type) : low);
		bounds.SetValue(key_idx, 1, stats.null_count > 0 && (!nulls_first || !has_lowest) ? Value(type) : high);
	}
	bounds.SetCardinality(2);
	Vector sort_keys(LogicalType::BLOB, 2);
	CreateSortKeyHelpers::CreateSortKey(bounds, modifiers, sort_keys);
	auto keys = FlatVector::GetData<string_t>(sort_keys);
	input.first_key = keys[0].GetString();
	input.last_key = keys[1].GetString();
	input.has_last_key = true;
	return true;
}

void DuckLakeSortedMergeSequencer::Initialize() {
	if (initialized) {
		return;
	}
	// the catalog's min/max statistics bound the keys of files sorted on plain columns, other files are read
	map<DataFileIndex, map<idx_t, DuckLakeFileKeyStats>> key_stats;
	vector<FieldIndex> fields;
	if (!decodable_columns.empty()) {
		for (auto &column_idx : decodable_columns) {
			fields.push_back(table.GetFieldId(PhysicalIndex(column_idx)).GetFieldIndex());
		}
		auto &transaction = DuckLakeTransaction::Get(context, table.ParentCatalog());
		key_stats = transaction.GetMetadataManager().ReadDataFileKeyStats(table.GetTableId(), transaction.GetSnapshot(),
		                                                                  fields);
	}
	vector<idx_t> by_first_key;
	for (auto &input : merge_inputs) {
		auto file_stats = key_stats.find(inputs[input->input_idx]->file.file_id);
		if (file_stats != key_stats.end() && CatalogBounds(file_stats->second, fields, *input)) {
			by_first_key.push_back(input->input_idx);
			continue;
		}
		input->started = true;
		if (!TakeChunk(*input)) {
			continue;
		}
		input->first_key = input->KeyAt(0).key.GetString();
		ReadLastKey(*input);
		by_first_key.push_back(input->input_idx);
	}
	std::sort(by_first_key.begin(), by_first_key.end(), [&](idx_t lhs, idx_t rhs) {
		auto lhs_key = KeyOf(merge_inputs[lhs]->first_key);
		auto rhs_key = KeyOf(merge_inputs[rhs]->first_key);
		return lhs_key < rhs_key || (lhs_key == rhs_key && lhs < rhs);
	});
	// an input starts a new section when it starts after every input before it ends, so sections are merged one after
	// the other. Equal keys keep inputs in one section, where ties are ordered by input.
	string section_end;
	bool section_end_known = false;
	for (auto input_idx : by_first_key) {
		auto &input = *merge_inputs[input_idx];
		if (sections.empty() || (section_end_known && KeyOf(section_end) < KeyOf(input.first_key))) {
			sections.emplace_back();
			section_end_known = true;
			section_end = input.last_key;
		}
		input.section = sections.size() - 1;
		sections.back().push_back(input_idx);
		if (!input.has_last_key) {
			section_end_known = false;
		} else if (section_end_known && KeyOf(section_end) < KeyOf(input.last_key)) {
			section_end = input.last_key;
		}
	}
	for (auto &input : merge_inputs) {
		if (input->exhausted) {
			input->section = NumericLimits<idx_t>::Maximum();
		}
	}
	for (auto &section : sections) {
		if (section.size() == 1 && inputs[section[0]]->known_sorted) {
			MakeKeyless(*merge_inputs[section[0]]);
		}
	}
	initialized = true;
}

void DuckLakeSortedMergeSequencer::MakeKeyless(MergeInput &input) {
	input.keyless = true;
	keyless_sections++;
	lock_guard<mutex> guard(input.segments_lock);
	if (input.segments.empty()) {
		return;
	}
	// read the rest of the segments already started again, without keys
	auto row_start = input.window.back()->row_numbers.back() + 1;
	auto row_end = input.segments.back()->row_end;
	segments_ahead -= input.segments.size();
	input.segments.clear();
	input.segments.push_back(make_shared_ptr<KeySegment>(row_start, row_end, true));
	segments_ahead++;
}

bool DuckLakeSortedMergeSequencer::StartNextSection() {
	while (section_inputs.empty() && next_section < sections.size()) {
		active_section = next_section;
		for (auto input_idx : sections[next_section]) {
			auto &input = *merge_inputs[input_idx];
			if (!input.started) {
				input.started = true;
				TakeChunk(input);
			}
			if (!input.exhausted) {
				section_inputs.push_back(input_idx);
			}
		}
		// ties between inputs are ordered by input
		std::sort(section_inputs.begin(), section_inputs.end());
		next_section++;
	}
	return !section_inputs.empty();
}

bool DuckLakeSortedMergeSequencer::InputIsNear(idx_t input_idx) const {
	// sections are known once the sequencer is initialized
	return !initialized || merge_inputs[input_idx]->section <= active_section + 1;
}

void DuckLakeSortedMergeSequencer::LoadKeys(MergeInput &input, KeySegment &segment) {
	auto &column_ids = segment.keyless ? row_number_column_ids : scan_column_ids;
	if (!segment.scanner) {
		unique_ptr<TableFilterSet> filters;
		if (segment.row_start > 0 || segment.row_end != NumericLimits<idx_t>::Maximum()) {
			filters = make_uniq<TableFilterSet>();
			ProjectionIndex row_number_column(column_ids.size() - 1);
			filters->PushFilter(row_number_column, ExpressionFilter::CreateComparisonFilter(
			                                           ExpressionType::COMPARE_GREATERTHANOREQUALTO,
			                                           Value::BIGINT(NumericCast<int64_t>(segment.row_start))));
			if (segment.row_end != NumericLimits<idx_t>::Maximum()) {
				filters->PushFilter(row_number_column, ExpressionFilter::CreateComparisonFilter(
				                                           ExpressionType::COMPARE_LESSTHAN,
				                                           Value::BIGINT(NumericCast<int64_t>(segment.row_end))));
			}
		}
		segment.scanner = make_uniq<DuckLakeSingleFileScanner>(context, table, *inputs[input.input_idx], column_ids,
		                                                       std::move(filters));
		segment.scan_chunk.Initialize(context, segment.scanner->GetTypes());
		if (!segment.keyless) {
			segment.key_chunk.Initialize(context, key_types);
			segment.executor = make_uniq<ExpressionExecutor>(context, key_expressions);
		}
	}
	if (!segment.scanner->Scan(segment.scan_chunk)) {
		segment.scanner.reset();
		segment.done = true;
		return;
	}
	auto count = segment.scan_chunk.size();
	auto chunk = make_shared_ptr<KeyChunk>();
	if (!segment.keyless) {
		segment.key_chunk.Reset();
		segment.executor->Execute(segment.scan_chunk, segment.key_chunk);
		chunk->sort_keys = make_shared_ptr<Vector>(LogicalType::BLOB, count);
		CreateSortKeyHelpers::CreateSortKey(segment.key_chunk, modifiers, *chunk->sort_keys);
		chunk->keys.reserve(count);
		for (auto entry : chunk->sort_keys->Values<string_t>()) {
			chunk->keys.emplace_back(entry.GetValue());
		}
	}
	auto &row_number_vector = segment.scan_chunk.data[segment.scan_chunk.ColumnCount() - 1];
	chunk->row_numbers.reserve(count);
	for (auto entry : row_number_vector.Values<int64_t>()) {
		chunk->row_numbers.push_back(NumericCast<idx_t>(entry.GetValue()));
	}
	auto last = chunk->row_numbers.back() + 1 == segment.row_end;
	{
		lock_guard<mutex> guard(segment.chunks_lock);
		segment.chunks.push_back(std::move(chunk));
		segment.loaded_chunks++;
	}
	if (last) {
		// the last row of the segment is loaded, release the scanner without another scan
		segment.scanner.reset();
		segment.done = true;
	}
}

bool DuckLakeSortedMergeSequencer::TakeChunk(MergeInput &input) {
	if (input.fully_loaded) {
		return false;
	}
	shared_ptr<KeyChunk> chunk;
	while (!chunk) {
		shared_ptr<KeySegment> segment;
		{
			lock_guard<mutex> guard(input.segments_lock);
			if (input.segments.empty()) {
				if (input.next_segment == input.segment_count) {
					input.fully_loaded = true;
					input.exhausted = input.loaded == 0;
					return false;
				}
				input.AddSegment();
				segments_ahead++;
			}
			segment = input.segments.front();
		}
		chunk = segment->TakeLoaded();
		if (!chunk && !segment->done) {
			// no chunk is loaded yet: load one, or wait for the worker loading one
			lock_guard<mutex> load_guard(segment->load_lock);
			chunk = segment->TakeLoaded();
			if (!chunk && !segment->done) {
				LoadKeys(input, *segment);
				chunk = segment->TakeLoaded();
			}
		}
		if (!chunk && segment->done) {
			// the last chunk can be loaded between taking chunks and seeing the segment done
			chunk = segment->TakeLoaded();
			if (!chunk) {
				// every row of the segment is taken, or deleted
				lock_guard<mutex> guard(input.segments_lock);
				input.segments.pop_front();
				segments_ahead--;
			}
		}
		if (chunk) {
			// wake a worker to load more keys
			NotifyWaiters();
		}
	}
	VerifySorted(input, *chunk);
	if (!chunk->keys.empty()) {
		input.previous_key_data = chunk->keys.back().key.GetString();
		input.previous_key = KeyOf(input.previous_key_data);
		input.has_previous_key = true;
	}
	auto previous_end = input.window_ends.empty() ? 0 : input.window_ends.back();
	input.window_ends.push_back(previous_end + chunk->row_numbers.size());
	input.loaded += chunk->row_numbers.size();
	input.window.push_back(std::move(chunk));
	return true;
}

void DuckLakeSortedMergeSequencer::VerifySorted(const MergeInput &input, const KeyChunk &chunk) {
	if (chunk.keys.empty()) {
		// the input is merged alone, only its row numbers are read
		return;
	}
	auto known_sorted = inputs[input.input_idx]->known_sorted;
#ifndef DEBUG
	// trust the catalog, debug builds verify it
	if (known_sorted) {
		return;
	}
#endif
	auto &keys = chunk.keys;
	bool out_of_order = input.has_previous_key && keys[0] < input.previous_key;
	for (idx_t i = 1; i < keys.size() && !out_of_order; i++) {
		out_of_order = keys[i] < keys[i - 1];
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
}

void DuckLakeSortedMergeSequencer::TakeRows(MergeInput &input, idx_t count, DuckLakeSequencedBatch::Range &range) {
	if (range.chunks.empty()) {
		range.input_idx = input.input_idx;
		range.offset = input.position;
	}
	range.row_count += count;
	input.loaded -= count;
	while (count > 0) {
		auto &front = input.window.front();
		auto take = MinValue(count, front->row_numbers.size() - input.position);
		if (range.chunks.empty() || range.chunks.back() != front) {
			range.chunks.push_back(front);
		}
		count -= take;
		input.position += take;
		if (input.position == front->row_numbers.size()) {
			// every row of the chunk is sequenced
			auto chunk_rows = front->row_numbers.size();
			input.window.pop_front();
			input.window_ends.erase(input.window_ends.begin());
			for (auto &end : input.window_ends) {
				end -= chunk_rows;
			}
			input.position = 0;
		}
	}
	if (input.loaded == 0 && !TakeChunk(input)) {
		input.exhausted = true;
	}
}

bool DuckLakeSortedMergeSequencer::TopUpSegment(MergeInput &input) {
	vector<shared_ptr<KeySegment>> candidates;
	{
		lock_guard<mutex> guard(input.segments_lock);
		for (auto &segment : input.segments) {
			if (!segment->done && segment->loaded_chunks < chunks_ahead) {
				candidates.push_back(segment);
			}
		}
	}
	for (auto &segment : candidates) {
		unique_lock<mutex> segment_guard(segment->load_lock, std::try_to_lock);
		if (segment_guard.owns_lock() && !segment->done && segment->loaded_chunks < chunks_ahead) {
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
		if (!InputIsNear(input->input_idx)) {
			// the input is merged in a later section
			continue;
		}
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
	unique_lock<mutex> segment_guard(segment->load_lock, std::try_to_lock);
	if (segment_guard.owns_lock() && !segment->done && segment->loaded_chunks == 0) {
		LoadKeys(*target, *segment);
	}
	return true;
}

bool DuckLakeSortedMergeSequencer::MergingKeys() const {
	if (!initialized) {
		return true;
	}
	auto section = active_section.load();
	return section < sections.size() && sections[section].size() > 1;
}

void DuckLakeSortedMergeSequencer::WaitForWork(idx_t seen_generation) {
	unique_lock<mutex> guard(waiters_lock);
	waiters.wait_for(guard, std::chrono::milliseconds(10), [&]() { return wait_generation != seen_generation; });
}

void DuckLakeSortedMergeSequencer::NotifyWaiters(bool all) {
	{
		lock_guard<mutex> guard(waiters_lock);
		wait_generation++;
	}
	if (all) {
		waiters.notify_all();
	} else {
		waiters.notify_one();
	}
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

optional_idx DuckLakeSortedMergeSequencer::NextRow(idx_t input_idx) const {
	auto &input = *merge_inputs[input_idx];
	if (input.exhausted) {
		return optional_idx();
	}
	return input.RowNumberAt(0);
}

void DuckLakeSortedMergeSequencer::SetBatchRows(idx_t batch_rows) {
	// read about a batch's share of every input of the widest section ahead
	idx_t widest = 1;
	for (auto &section : sections) {
		widest = MaxValue<idx_t>(widest, section.size());
	}
	auto chunks = (batch_rows + STANDARD_VECTOR_SIZE * widest - 1) / (STANDARD_VECTOR_SIZE * widest);
	chunks_ahead = MaxValue<idx_t>(MIN_CHUNKS_AHEAD, MinValue<idx_t>(MAX_CHUNKS_AHEAD, chunks + 2));
}

//! The loaded rows of a section's inputs, ordered by key and then by input, for the multisequence selection
struct DuckLakeSortedMergeSequencer::WindowSequences {
	vector<const MergeInput *> sequences;

	idx_t Count() const {
		return sequences.size();
	}
	bool Less(idx_t lhs_sequence, idx_t lhs_position, idx_t rhs_sequence, idx_t rhs_position) const {
		auto &lhs = sequences[lhs_sequence]->KeyAt(lhs_position);
		auto &rhs = sequences[rhs_sequence]->KeyAt(rhs_position);
		return lhs < rhs || (lhs_sequence < rhs_sequence && lhs == rhs);
	}
};

idx_t DuckLakeSortedMergeSequencer::RowsUpTo(const MergeInput &frontier) const {
	auto &bound = frontier.KeyAt(frontier.loaded - 1);
	idx_t rows = 0;
	for (auto input_idx : section_inputs) {
		auto &input = *merge_inputs[input_idx];
		if (&input == &frontier) {
			rows += input.loaded;
			continue;
		}
		// rows of earlier inputs come before equal keys of the frontier, rows of later inputs after them
		auto inclusive = input.input_idx < frontier.input_idx;
		idx_t lo = 0;
		idx_t hi = input.loaded;
		while (lo < hi) {
			auto middle = lo + (hi - lo) / 2;
			auto &key = input.KeyAt(middle);
			if (key < bound || (inclusive && key == bound)) {
				lo = middle + 1;
			} else {
				hi = middle;
			}
		}
		rows += lo;
	}
	return rows;
}

void DuckLakeSortedMergeSequencer::SequenceSection(idx_t max_rows, vector<DuckLakeSequencedBatch::Range> &part) {
	// the batch is the first max_rows rows of the section's loaded rows, as long as those all come before the first
	// row not loaded yet: take more keys from the input whose last loaded key comes first until they do
	while (true) {
		optional_ptr<MergeInput> frontier;
		for (auto input_idx : section_inputs) {
			auto &input = *merge_inputs[input_idx];
			if (input.fully_loaded) {
				continue;
			}
			if (!frontier || input.KeyAt(input.loaded - 1) < frontier->KeyAt(frontier->loaded - 1)) {
				frontier = &input;
			}
		}
		if (!frontier || RowsUpTo(*frontier) >= max_rows) {
			break;
		}
		TakeChunk(*frontier);
	}
	WindowSequences window_sequences;
	vector<idx_t> starts;
	vector<idx_t> ends;
	for (auto input_idx : section_inputs) {
		auto &input = *merge_inputs[input_idx];
		window_sequences.sequences.push_back(&input);
		starts.push_back(0);
		ends.push_back(input.loaded);
	}
	vector<idx_t> cuts;
	MultisequenceSelection<WindowSequences>::Select(window_sequences, starts, ends, max_rows, cuts);
	for (idx_t i = 0; i < section_inputs.size(); i++) {
		if (cuts[i] == 0) {
			continue;
		}
		auto &range = part.emplace_back();
		TakeRows(*merge_inputs[section_inputs[i]], cuts[i], range);
	}
}

bool DuckLakeSortedMergeSequencer::NextBatch(idx_t max_rows, DuckLakeSequencedBatch &batch) {
	if (!initialized) {
		Initialize();
	}
	batch.Clear();
	while (batch.row_count < max_rows && (!section_inputs.empty() || StartNextSection())) {
		auto remaining = max_rows - batch.row_count;
		auto &part = batch.parts.emplace_back();
		if (section_inputs.size() == 1) {
			// the rows of one input are already in order
			auto &input = *merge_inputs[section_inputs[0]];
			auto &range = part.emplace_back();
			while (remaining > 0 && !input.exhausted) {
				auto take = MinValue(remaining, input.loaded);
				TakeRows(input, take, range);
				remaining -= take;
			}
		} else {
			SequenceSection(remaining, part);
		}
		for (auto &range : part) {
			batch.row_count += range.row_count;
		}
		// inputs without rows left leave the section
		section_inputs.erase(std::remove_if(section_inputs.begin(), section_inputs.end(),
		                                    [&](idx_t input_idx) { return merge_inputs[input_idx]->exhausted; }),
		                     section_inputs.end());
	}
	return batch.row_count > 0;
}

void DuckLakeSortedMergeSequencer::BatchInputRows(const DuckLakeSequencedBatch &batch,
                                                  vector<DuckLakeMergeSlice> &input_rows) {
	input_rows.clear();
	for (auto &part : batch.parts) {
		for (auto &range : part) {
			// find the range's last row
			idx_t chunk_idx = 0;
			idx_t row = range.offset;
			idx_t remaining = range.row_count;
			while (remaining > range.chunks[chunk_idx]->row_numbers.size() - row) {
				remaining -= range.chunks[chunk_idx]->row_numbers.size() - row;
				chunk_idx++;
				row = 0;
			}
			auto row_start = range.chunks.front()->row_numbers[range.offset];
			auto row_end = range.chunks[chunk_idx]->row_numbers[row + remaining - 1] + 1;
			input_rows.push_back(DuckLakeMergeSlice {range.input_idx, row_start, row_end, range.row_count});
		}
	}
}

//! Reads the rows of a range in order
struct RangeReader {
	explicit RangeReader(const DuckLakeSequencedBatch::Range &range)
	    : range(range), chunk(range.chunks[0].get()), chunk_idx(0), row(range.offset), remaining(range.row_count) {
		CacheKey();
	}

	const DuckLakeSequencedBatch::Range &range;
	//! The chunk of the current row
	const KeyChunk *chunk;
	idx_t chunk_idx;
	idx_t row;
	idx_t remaining;
	//! The leading words of the current key, so most comparisons do not leave the readers
	uint64_t high = 0;
	uint64_t low = 0;

	void CacheKey() {
		if (remaining > 0 && !Chunk().keys.empty()) {
			high = Key().high;
			low = Key().low;
		}
	}

	const KeyChunk &Chunk() const {
		return *chunk;
	}
	const MergeKey &Key() const {
		return Chunk().keys[row];
	}
	idx_t RowNumber() const {
		return Chunk().row_numbers[row];
	}
	void Advance(idx_t count = 1) {
		remaining -= count;
		row += count;
		if (remaining > 0 && row == chunk->row_numbers.size()) {
			chunk_idx++;
			chunk = range.chunks[chunk_idx].get();
			row = 0;
		}
		CacheKey();
	}
	//! The rows of the current chunk from the current row on that come before bound, at least one
	idx_t RunBefore(const MergeKey &bound, bool inclusive) const {
		auto &keys = Chunk().keys;
		auto end = MinValue(row + remaining, keys.size());
		auto before = [&](idx_t i) {
			return keys[i] < bound || (inclusive && keys[i] == bound);
		};
		// gallop to bracket the first row that does not come before bound, then binary search
		idx_t lo = row + 1;
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
			auto middle = lo + (hi - lo) / 2;
			if (before(middle)) {
				lo = middle + 1;
			} else {
				hi = middle;
			}
		}
		return lo - row;
	}
};

//! Appends rows to the slices and sort keys of a batch in merged order
struct BatchOutput {
	BatchOutput(vector<DuckLakeMergeSlice> &slices, optional_ptr<DuckLakeMergeKeys> keys) : slices(slices), keys(keys) {
	}

	vector<DuckLakeMergeSlice> &slices;
	optional_ptr<DuckLakeMergeKeys> keys;

	//! Append the rows of a range that are already in order, a chunk at a time
	void AppendRange(const DuckLakeSequencedBatch::Range &range) {
		idx_t row = range.offset;
		idx_t remaining = range.row_count;
		for (idx_t chunk_idx = 0; remaining > 0; chunk_idx++) {
			auto &chunk = *range.chunks[chunk_idx];
			auto count = MinValue(remaining, chunk.row_numbers.size() - row);
			auto has_keys = !chunk.keys.empty();
			if (slices.empty() || slices.back().input_idx != range.input_idx || slices.back().has_keys != has_keys) {
				DuckLakeMergeSlice slice {range.input_idx, chunk.row_numbers[row], 0, 0};
				slice.has_keys = has_keys;
				slices.push_back(slice);
			}
			slices.back().row_end = chunk.row_numbers[row + count - 1] + 1;
			slices.back().row_count += count;
			if (keys && has_keys) {
				for (idx_t i = row; i < row + count; i++) {
					keys->keys.push_back(chunk.keys[i].key);
				}
			}
			remaining -= count;
			row = 0;
		}
	}

	//! Append the next count rows of the reader, which are in its current chunk
	void Append(const RangeReader &reader, idx_t count = 1) {
		auto &chunk = reader.Chunk();
		auto has_keys = !chunk.keys.empty();
		if (slices.empty() || slices.back().input_idx != reader.range.input_idx || slices.back().has_keys != has_keys) {
			DuckLakeMergeSlice slice {reader.range.input_idx, reader.RowNumber(), 0, 0};
			slice.has_keys = has_keys;
			slices.push_back(slice);
		}
		slices.back().row_end = chunk.row_numbers[reader.row + count - 1] + 1;
		slices.back().row_count += count;
		if (keys && has_keys) {
			for (idx_t i = reader.row; i < reader.row + count; i++) {
				keys->keys.push_back(chunk.keys[i].key);
			}
		}
	}
};

//! Merges the ranges of a section with a loser tree: each node holds the reader that lost there, the root the winner
static void MergeRanges(const vector<DuckLakeSequencedBatch::Range> &ranges, BatchOutput &output) {
	vector<RangeReader> readers;
	for (auto &range : ranges) {
		readers.emplace_back(range);
	}
	auto wins = [&](idx_t lhs, idx_t rhs) {
		if (lhs == DConstants::INVALID_INDEX || readers[lhs].remaining == 0) {
			return false;
		}
		if (rhs == DConstants::INVALID_INDEX || readers[rhs].remaining == 0) {
			return true;
		}
		auto &lhs_reader = readers[lhs];
		auto &rhs_reader = readers[rhs];
		if (lhs_reader.high != rhs_reader.high) {
			return lhs_reader.high < rhs_reader.high;
		}
		if (lhs_reader.low != rhs_reader.low) {
			return lhs_reader.low < rhs_reader.low;
		}
		auto &lhs_key = lhs_reader.Key();
		auto &rhs_key = rhs_reader.Key();
		return lhs_key < rhs_key || (readers[lhs].range.input_idx < readers[rhs].range.input_idx && lhs_key == rhs_key);
	};
	idx_t leaves = 1;
	while (leaves < readers.size()) {
		leaves *= 2;
	}
	vector<idx_t> winners(2 * leaves, DConstants::INVALID_INDEX);
	for (idx_t i = 0; i < readers.size(); i++) {
		winners[leaves + i] = i;
	}
	vector<idx_t> tree(leaves, DConstants::INVALID_INDEX);
	for (idx_t node = leaves - 1; node >= 1; node--) {
		auto lhs = winners[2 * node];
		auto rhs = winners[2 * node + 1];
		winners[node] = wins(rhs, lhs) ? rhs : lhs;
		tree[node] = wins(rhs, lhs) ? lhs : rhs;
	}
	// while the readers interleave, take one row and replay the winner's path. Once the winner wins again, take its run
	// of rows before the runner-up, the best of the readers that lost on its path.
	auto winner = winners[1];
	auto previous = DConstants::INVALID_INDEX;
	while (winner != DConstants::INVALID_INDEX && readers[winner].remaining > 0) {
		auto &reader = readers[winner];
		auto leaf = leaves + winner;
		idx_t count = 1;
		if (winner == previous) {
			auto runner_up = DConstants::INVALID_INDEX;
			for (auto node = leaf / 2; node >= 1; node /= 2) {
				if (wins(tree[node], runner_up)) {
					runner_up = tree[node];
				}
			}
			if (runner_up == DConstants::INVALID_INDEX || readers[runner_up].remaining == 0) {
				count = MinValue(reader.remaining, reader.Chunk().row_numbers.size() - reader.row);
			} else {
				auto inclusive = reader.range.input_idx < readers[runner_up].range.input_idx;
				count = reader.RunBefore(readers[runner_up].Key(), inclusive);
			}
		}
		output.Append(reader, count);
		reader.Advance(count);
		previous = winner;
		for (auto node = leaf / 2; node >= 1; node /= 2) {
			if (wins(tree[node], winner)) {
				std::swap(tree[node], winner);
			}
		}
	}
}

void DuckLakeSortedMergeSequencer::OrderBatch(const DuckLakeSequencedBatch &batch, vector<DuckLakeMergeSlice> &slices,
                                              optional_ptr<DuckLakeMergeKeys> keys) {
	slices.clear();
	if (keys) {
		// the keys point into the sort keys of the batch's chunks
		for (auto &part : batch.parts) {
			for (auto &range : part) {
				for (auto &chunk : range.chunks) {
					if (chunk->sort_keys) {
						keys->owners.push_back(chunk->sort_keys);
					}
				}
			}
		}
	}
	BatchOutput output(slices, keys);
	for (auto &part : batch.parts) {
		if (part.size() == 1) {
			output.AppendRange(part[0]);
		} else if (part.size() > 1) {
			MergeRanges(part, output);
		}
	}
}

} // namespace duckdb
