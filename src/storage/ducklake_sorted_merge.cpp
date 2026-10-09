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

//! The current key of an input in the loser tree, kept in one array so comparisons do not chase pointers
struct DuckLakeSortedMergeSequencer::CurrentKey {
	uint64_t high = 0;
	uint64_t low = 0;
	bool exhausted = true;
};

//! The sort keys of a chunk of an input, in file order, or only its row numbers when its keys are not needed
struct DuckLakeSortedMergeSequencer::KeyChunk {
	shared_ptr<Vector> sort_keys;
	vector<MergeKey> keys;
	vector<idx_t> row_numbers;
};

//! The keys of file rows [row_start, row_end) of an input, loaded a chunk at a time by whichever thread gets to it
struct DuckLakeSortedMergeSequencer::KeySegment {
	KeySegment(idx_t row_start, idx_t row_end, bool keyless = false)
	    : row_start(row_start), row_end(row_end), keyless(keyless) {
	}

	idx_t row_start;
	idx_t row_end;
	//! Whether only row numbers are read
	bool keyless;
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
	//! The first and last key of the input, the last unknown when its last row is deleted or its row count is not known
	string first_key;
	string last_key;
	bool has_last_key = false;
	//! The section of inputs whose key ranges overlap that the input is merged in
	idx_t section = 0;
	//! Whether the input is merged alone and recorded as sorted, so its keys are not needed
	bool keyless = false;
	//! Whether the first chunk is loaded, which waits for the input's section when the catalog bounds its keys
	bool loaded = false;
	//! The input's leaf in the loser tree of its section
	idx_t leaf = 0;

	const MergeKey &CurrentKey() const {
		return keys[position];
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
	current_keys.resize(merge_inputs.size());
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
		input->loaded = true;
		if (!LoadChunk(*input)) {
			continue;
		}
		input->first_key = input->keys[0].key.GetString();
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
	auto row_start = input.row_numbers.back() + 1;
	auto row_end = input.segments.back()->row_end;
	segments_ahead -= input.segments.size();
	input.segments.clear();
	input.segments.push_back(make_shared_ptr<KeySegment>(row_start, row_end, true));
	segments_ahead++;
}

bool DuckLakeSortedMergeSequencer::StartNextSection() {
	while (!SectionActive() && next_section < sections.size()) {
		active_section = next_section;
		vector<idx_t> section_inputs;
		for (auto input_idx : sections[next_section]) {
			auto &input = *merge_inputs[input_idx];
			if (!input.loaded) {
				input.loaded = true;
				LoadChunk(input);
			}
			if (!input.exhausted) {
				RefreshKey(input);
				section_inputs.push_back(input_idx);
			}
		}
		BuildTree(section_inputs);
		next_section++;
	}
	return SectionActive();
}

bool DuckLakeSortedMergeSequencer::SectionActive() const {
	return !tree.empty() && tree[0] != DConstants::INVALID_INDEX && !merge_inputs[tree[0]]->exhausted;
}

bool DuckLakeSortedMergeSequencer::Wins(idx_t lhs, idx_t rhs) const {
	if (lhs == DConstants::INVALID_INDEX || current_keys[lhs].exhausted) {
		return false;
	}
	if (rhs == DConstants::INVALID_INDEX || current_keys[rhs].exhausted) {
		return true;
	}
	auto &lhs_key = current_keys[lhs];
	auto &rhs_key = current_keys[rhs];
	if (lhs_key.high != rhs_key.high) {
		return lhs_key.high < rhs_key.high;
	}
	if (lhs_key.low != rhs_key.low) {
		return lhs_key.low < rhs_key.low;
	}
	return InputBefore(*merge_inputs[lhs], *merge_inputs[rhs]);
}

void DuckLakeSortedMergeSequencer::RefreshKey(const MergeInput &input) {
	auto &current = current_keys[input.input_idx];
	current.exhausted = input.exhausted || input.position >= input.keys.size();
	if (!current.exhausted) {
		auto &key = input.keys[input.position];
		current.high = key.high;
		current.low = key.low;
	}
}

void DuckLakeSortedMergeSequencer::BuildTree(const vector<idx_t> &section_inputs) {
	tree.clear();
	if (section_inputs.empty()) {
		return;
	}
	idx_t leaves = 1;
	while (leaves < section_inputs.size()) {
		leaves *= 2;
	}
	// winners of the subtrees, the leaves at [leaves, 2 * leaves)
	vector<idx_t> winners(2 * leaves, DConstants::INVALID_INDEX);
	for (idx_t i = 0; i < section_inputs.size(); i++) {
		winners[leaves + i] = section_inputs[i];
		merge_inputs[section_inputs[i]]->leaf = leaves + i;
	}
	tree.assign(leaves, DConstants::INVALID_INDEX);
	for (idx_t node = leaves - 1; node >= 1; node--) {
		auto lhs = winners[2 * node];
		auto rhs = winners[2 * node + 1];
		winners[node] = Wins(rhs, lhs) ? rhs : lhs;
		tree[node] = Wins(rhs, lhs) ? lhs : rhs;
	}
	tree[0] = winners[1];
}

void DuckLakeSortedMergeSequencer::Replay(const MergeInput &input) {
	auto winner = input.input_idx;
	for (auto node = input.leaf / 2; node >= 1; node /= 2) {
		if (Wins(tree[node], winner)) {
			std::swap(tree[node], winner);
		}
	}
	tree[0] = winner;
}

optional_ptr<DuckLakeSortedMergeSequencer::MergeInput>
DuckLakeSortedMergeSequencer::RunnerUp(const MergeInput &winner) const {
	// every other input lost to one of the inputs on the winner's path
	auto runner_up = DConstants::INVALID_INDEX;
	for (auto node = winner.leaf / 2; node >= 1; node /= 2) {
		if (Wins(tree[node], runner_up)) {
			runner_up = tree[node];
		}
	}
	if (runner_up == DConstants::INVALID_INDEX || merge_inputs[runner_up]->exhausted) {
		return nullptr;
	}
	return merge_inputs[runner_up].get();
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
	auto chunk = make_uniq<KeyChunk>();
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
			// wake a worker to load more keys
			NotifyWaiters();
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
	if (input.keys.empty()) {
		// the input is merged alone, only its row numbers are read
		return true;
	}

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
	unique_lock<mutex> segment_guard(segment->lock, std::try_to_lock);
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

void DuckLakeSortedMergeSequencer::NotifyWaiters() {
	{
		lock_guard<mutex> guard(waiters_lock);
		wait_generation++;
	}
	waiters.notify_one();
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
	while (sequenced < max_rows && (SectionActive() || StartNextSection())) {
		auto &input = *merge_inputs[tree[0]];
		DuckLakeMergeSlice slice {input.input_idx, input.row_numbers[input.position], 0, 0};
		slice.has_keys = !input.keys.empty();
		// the winner's current row comes first. While the inputs interleave, take one row and replay the winner's path;
		// once the winner wins again, find the run of its rows before the runner-up.
		bool won_again = false;
		bool replayed = false;
		while (true) {
			auto end = MinValue(input.row_numbers.size(), input.position + max_rows - sequenced);
			auto run_end = input.position + 1;
			if (won_again) {
				run_end = RunEnd(input, MinValue(input.position + 1, end), end, RunnerUp(input));
			}
			if (keys && slice.has_keys) {
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
			replayed = false;
			if (input.position == input.row_numbers.size() &&
			    (!LoadChunk(input) || input.keys.empty() == slice.has_keys)) {
				// the input is exhausted, or a slice has the keys of all its rows or of none
				RefreshKey(input);
				break;
			}
			RefreshKey(input);
			if (sequenced == max_rows) {
				break;
			}
			Replay(input);
			replayed = true;
			if (tree[0] != input.input_idx) {
				break;
			}
			won_again = true;
		}
		D_ASSERT(slice.row_count > 0);
		slices.push_back(slice);
		if (!replayed) {
			Replay(input);
		}
	}
	return !slices.empty();
}

} // namespace duckdb
