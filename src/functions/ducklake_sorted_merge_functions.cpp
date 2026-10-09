#include "functions/ducklake_table_functions.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/partition_info.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "storage/ducklake_catalog.hpp"
#include "storage/ducklake_sorted_merge.hpp"
#include "storage/ducklake_table_entry.hpp"
#include "storage/ducklake_transaction.hpp"

namespace duckdb {

static void AddSortedMergeArguments(FunctionSignature &signature, bool add_summary = false) {
	signature.AddPositionalOnly("table_name", LogicalType::VARCHAR)
	    .WithTypedKwargs("options", [&](TypedKwargs &options) {
		    options.Add("schema", LogicalType::VARCHAR).Add("row_group_size", LogicalType::BIGINT);
		    if (add_summary) {
			    options.Add("summary", LogicalType::BOOLEAN);
		    } else {
			    options.Add("decode_unit_size", LogicalType::BIGINT);
		    }
	    });
}

static DuckLakeTableEntry &BindSortedMergeTable(ClientContext &context, TableFunctionBindInput &input,
                                                vector<DuckLakeFileListEntry> &files, idx_t &row_group_size) {
	auto &catalog = DuckLakeBaseMetadataFunction::GetCatalog(context, input);
	auto &transaction = DuckLakeTransaction::Get(context, catalog);
	auto schema = DuckLakeTableFunctionUtil::GetStringOption(input, "schema");
	auto table_name = DuckLakeTableFunctionUtil::GetTableName(input.inputs[1]);
	auto &table = DuckLakeBaseMetadataFunction::GetTableEntry(context, catalog, schema, table_name);
	if (!DuckLakeSortedMergePlanner::HasMergeableSort(table)) {
		throw InvalidInputException("Table \"%s\" has no sort key to merge on", table_name);
	}
	row_group_size = DEFAULT_ROW_GROUP_SIZE;
	Value row_group_size_value;
	if (DuckLakeTableFunctionUtil::TryGetNonNullOption(input, "row_group_size", "integer", row_group_size_value)) {
		auto value = row_group_size_value.GetValue<int64_t>();
		if (value <= 0) {
			throw InvalidInputException("row_group_size must be positive");
		}
		row_group_size = NumericCast<idx_t>(value);
	}
	files = transaction.GetMetadataManager().GetFilesForTable(table, transaction.GetSnapshot(), nullptr);
	return table;
}

//! Row groups per segment of sort keys, each opened by its own scan
static constexpr idx_t ROW_GROUPS_PER_KEY_SEGMENT = 8;

static vector<unique_ptr<DuckLakeFileScanBind>> BindMergeInputs(ClientContext &context, DuckLakeTableEntry &table,
                                                                const vector<DuckLakeFileListEntry> &files) {
	auto &transaction = DuckLakeTransaction::Get(context, table.ParentCatalog());
	auto file_info =
	    transaction.GetMetadataManager().ReadDataFileMergeInfo(table.GetTableId(), transaction.GetSnapshot());
	auto current_sort_id = table.GetSortData()->WrittenSortId();
	vector<unique_ptr<DuckLakeFileScanBind>> result;
	for (auto &file : files) {
		auto input = make_uniq<DuckLakeFileScanBind>(context, table, file);
		auto entry = file_info.find(file.file_id);
		if (entry != file_info.end()) {
			auto &sort_id = entry->second.sort_id;
			input->known_sorted =
			    current_sort_id.IsValid() && sort_id.IsValid() && sort_id.GetIndex() == current_sort_id.GetIndex();
			input->row_count = entry->second.record_count;
		}
		result.push_back(std::move(input));
	}
	return result;
}

//===--------------------------------------------------------------------===//
// ducklake_sorted_merge_plan
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> DuckLakeSortedMergePlanBind(ClientContext &context, TableFunctionBindInput &input,
                                                            vector<LogicalType> &return_types,
                                                            vector<Identifier> &names) {
	vector<DuckLakeFileListEntry> files;
	idx_t row_group_size;
	auto &table = BindSortedMergeTable(context, input, files, row_group_size);
	Value summary_value;
	bool summary = DuckLakeTableFunctionUtil::TryGetNonNullOption(input, "summary", "boolean", summary_value) &&
	               BooleanValue::Get(summary_value);

	names.emplace_back("row_group");
	return_types.emplace_back(LogicalType::BIGINT);
	if (summary) {
		names.emplace_back("slices");
		return_types.emplace_back(LogicalType::BIGINT);
		names.emplace_back("inputs");
		return_types.emplace_back(LogicalType::BIGINT);
	} else {
		names.emplace_back("input_index");
		return_types.emplace_back(LogicalType::BIGINT);
		names.emplace_back("data_file");
		return_types.emplace_back(LogicalType::VARCHAR);
		names.emplace_back("row_start");
		return_types.emplace_back(LogicalType::BIGINT);
		names.emplace_back("row_end");
		return_types.emplace_back(LogicalType::BIGINT);
	}
	names.emplace_back("row_count");
	return_types.emplace_back(LogicalType::BIGINT);

	auto inputs = BindMergeInputs(context, table, files);
	DuckLakeSortedMergePlanner planner(context, table, inputs, row_group_size * ROW_GROUPS_PER_KEY_SEGMENT);
	auto result = make_uniq<MetadataBindData>();
	vector<DuckLakeMergeSlice> slices;
	for (idx_t row_group = 0; planner.NextBatch(row_group_size, slices); row_group++) {
		if (summary) {
			set<idx_t> touched_inputs;
			idx_t row_count = 0;
			for (auto &slice : slices) {
				touched_inputs.insert(slice.input_idx);
				row_count += slice.row_count;
			}
			vector<Value> row_values;
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(row_group)));
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(slices.size())));
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(touched_inputs.size())));
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(row_count)));
			result->rows.push_back(std::move(row_values));
			continue;
		}
		for (auto &slice : slices) {
			vector<Value> row_values;
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(row_group)));
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(slice.input_idx)));
			row_values.emplace_back(files[slice.input_idx].file.path);
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(slice.row_start)));
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(slice.row_end)));
			row_values.push_back(Value::BIGINT(NumericCast<int64_t>(slice.row_count)));
			result->rows.push_back(std::move(row_values));
		}
	}
	return std::move(result);
}

DuckLakeSortedMergePlanFunction::DuckLakeSortedMergePlanFunction()
    : DuckLakeBaseMetadataFunction("ducklake_sorted_merge_plan", DuckLakeSortedMergePlanBind) {
	AddSortedMergeArguments(GetSignature(), true);
}

//===--------------------------------------------------------------------===//
// ducklake_sorted_merge
//===--------------------------------------------------------------------===//
struct SortedMergeBindData {
	explicit SortedMergeBindData(DuckLakeTableEntry &table) : table(table) {
	}

	DuckLakeTableEntry &table;
	vector<DuckLakeFileListEntry> files;
	idx_t row_group_size;
	//! Decoded bytes per decode unit, derived from the memory limit when not set
	optional_idx decode_unit_size;
};

//! The live rows of input rows [row_start, row_end), decoded once and shared by every batch that reads them
struct SortedMergeDecodeUnit {
	idx_t row_start;
	idx_t row_end;
	unique_ptr<ColumnDataCollection> rows;
	//! The file row number of every row in rows
	vector<idx_t> row_numbers;
};

//! Reads one segment of an input front to back, cutting it into decode units of about a target size in decoded bytes
struct SortedMergeSegmentReader {
	SortedMergeSegmentReader(idx_t row_start, idx_t row_end) : row_end(row_end), next_unit_start(row_start) {
	}

	mutex lock;
	//! One past the last file row of the segment
	idx_t row_end;
	unique_ptr<DuckLakeSingleFileScanner> scanner;
	DataChunk scan_chunk;
	bool exhausted = false;
	//! The first file row of the next unit
	idx_t next_unit_start;
	//! Decoded units that a batch may still need
	deque<shared_ptr<SortedMergeDecodeUnit>> units;
};

//! The segments of one input, each read by its own reader so different parts of a file decode in parallel
struct SortedMergeInputReader {
	mutex lock;
	//! Segment readers by segment index
	map<idx_t, shared_ptr<SortedMergeSegmentReader>> segments;
	//! The first row of each admitted batch that has not yet taken its units
	multiset<idx_t> pending_batch_starts;
	//! Rows before this are planned, so no batch that is not yet admitted needs them
	idx_t planned_row = 0;
};

//! The inherited lock guards the batch counts and the parked workers
struct SortedMergeGlobalState : public GlobalSourceState {
	//! Guards the planner
	mutex plan_lock;
	vector<unique_ptr<DuckLakeFileScanBind>> inputs;
	unique_ptr<DuckLakeSortedMergePlanner> planner;
	vector<unique_ptr<SortedMergeInputReader>> readers;
	//! The batch index of the next batch the planner hands out
	atomic<idx_t> next_batch_index {0};
	idx_t max_threads = 1;
	//! Whether the sink orders rows by batch index, so a parked worker reports the batches it has finished
	bool requires_batch_index = false;
	//! Batches being emitted
	idx_t batches_in_flight = 0;
	//! Workers allowed from the decoded size of the batches, as the sink measures it
	unique_ptr<DuckLakeSortedMergeWorkerPolicy> worker_policy;
	bool planner_exhausted = false;
	//! For logging and EXPLAIN ANALYZE
	idx_t worker_limit = 0;
	//! The range of worker limits once a batch is measured
	idx_t min_worker_limit = NumericLimits<idx_t>::Maximum();
	idx_t max_worker_limit = 0;
	idx_t peak_batches_in_flight = 0;
	idx_t batches_emitted = 0;
	idx_t parks = 0;
	//! Units decoded by workers waiting to start a batch
	atomic<idx_t> units_decoded_ahead {0};
	//! Where the next decode ahead starts looking, so inputs are visited in turn
	atomic<idx_t> decode_ahead_cursor {0};
	//! Decoded bytes a unit grows to before the next one starts
	idx_t unit_size_bytes = 0;
	//! File rows per segment, segments are aligned to row groups of this size
	idx_t segment_size = 0;

	idx_t MaxThreads() override {
		return max_threads;
	}
};

//! Where the current batch reads the rows of one input
struct SortedMergeInputCursor {
	//! The first and one past the last file row the batch reads
	idx_t row_start = 0;
	idx_t row_end = 0;
	vector<shared_ptr<SortedMergeDecodeUnit>> units;
	idx_t unit_idx = 0;
	//! Row of the current unit to read next
	idx_t unit_row = 0;
	ColumnDataScanState scan_state;
	DataChunk chunk;
	bool positioned = false;
};

struct SortedMergeLocalState : public LocalSourceState {
	explicit SortedMergeLocalState(SortedMergeGlobalState &gstate) : gstate(gstate) {
	}
	//! A worker that stops inside a batch, e.g. when a LIMIT is reached, gives its slot back to the parked workers
	~SortedMergeLocalState() override;

	SortedMergeGlobalState &gstate;
	idx_t batch_index = 0;
	bool has_batch = false;
	//! Decoded bytes of the rows emitted for the current batch
	idx_t batch_bytes = 0;
	vector<DuckLakeMergeSlice> slices;
	idx_t slice_idx = 0;
	idx_t slice_offset = 0;
	unordered_map<idx_t, SortedMergeInputCursor> cursors;
	//! The columns read from the files: their positions in a decoded unit, types, table columns and output columns
	vector<column_t> payload_columns;
	vector<LogicalType> payload_types;
	vector<column_t> payload_scan_columns;
	vector<idx_t> payload_output_columns;
	//! The sort key columns, decoded from the batch's sort keys instead of read from the files
	vector<idx_t> key_output_columns;
	vector<OrderModifiers> key_modifiers;
	DataChunk decoded_keys;
	//! The sort keys of the current batch's rows in merged order, and how many are emitted
	unique_ptr<DuckLakeMergeKeys> batch_keys;
	idx_t key_offset = 0;
};

//! Units take at most this fraction of the memory limit across all inputs
static constexpr idx_t UNIT_MEMORY_FRACTION = 16;
static constexpr idx_t MIN_UNIT_SIZE_BYTES = 1ULL << 20;
static constexpr idx_t MAX_UNIT_SIZE_BYTES = 32ULL << 20;
//! Memory limit per segment of sort keys the planner may load ahead
static constexpr idx_t KEY_SEGMENT_MEMORY = 64ULL << 20;

static unique_ptr<SortedMergeGlobalState> InitSortedMergeGlobal(ClientContext &context,
                                                                const SortedMergeBindData &bind_data) {
	auto result = make_uniq<SortedMergeGlobalState>();
	result->inputs = BindMergeInputs(context, bind_data.table, bind_data.files);
	// every input keeps about one unit decoded, so size units to keep that a small part of the memory limit
	auto max_memory = BufferManager::GetBufferManager(context).GetMaxMemory();
	result->planner = make_uniq<DuckLakeSortedMergePlanner>(context, bind_data.table, result->inputs,
	                                                        bind_data.row_group_size * ROW_GROUPS_PER_KEY_SEGMENT);
	// load key segments ahead only when the memory limit leaves room for at least one per input, unless the merge
	// reads nothing but sort keys
	auto input_count = result->inputs.size();
	auto segments_ahead = max_memory / KEY_SEGMENT_MEMORY;
	auto keys_only = result->planner->DecodableColumns().size() == bind_data.table.GetColumns().PhysicalColumnCount();
	if (keys_only) {
		segments_ahead = MaxValue(segments_ahead, 2 * input_count);
	} else if (segments_ahead < 2 * input_count) {
		segments_ahead = input_count;
	}
	result->planner->SetMaxSegmentsAhead(segments_ahead);
	for (idx_t input_idx = 0; input_idx < result->inputs.size(); input_idx++) {
		result->readers.push_back(make_uniq<SortedMergeInputReader>());
	}
	auto unit_size = max_memory / (UNIT_MEMORY_FRACTION * MaxValue<idx_t>(result->inputs.size(), 1));
	result->unit_size_bytes = MaxValue(MIN_UNIT_SIZE_BYTES, MinValue(MAX_UNIT_SIZE_BYTES, unit_size));
	if (bind_data.decode_unit_size.IsValid()) {
		result->unit_size_bytes = bind_data.decode_unit_size.GetIndex();
	}
	result->segment_size = bind_data.row_group_size;
	result->max_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	result->worker_policy = make_uniq<DuckLakeSortedMergeWorkerPolicy>(result->max_threads);
	return std::move(result);
}

static unique_ptr<SortedMergeLocalState> InitSortedMergeLocal(const SortedMergeBindData &bind_data,
                                                              SortedMergeGlobalState &gstate) {
	auto result = make_uniq<SortedMergeLocalState>(gstate);
	auto &key_columns = gstate.planner->DecodableColumns();
	vector<LogicalType> key_types;
	for (auto &column_idx : key_columns) {
		result->key_output_columns.push_back(column_idx);
		key_types.push_back(bind_data.table.GetColumns().GetColumn(PhysicalIndex(column_idx)).Type());
	}
	idx_t column_idx = 0;
	for (auto &column : bind_data.table.GetColumns().Physical()) {
		if (std::find(key_columns.begin(), key_columns.end(), column_idx) == key_columns.end()) {
			result->payload_columns.push_back(result->payload_types.size());
			result->payload_types.push_back(column.Type());
			result->payload_scan_columns.push_back(column_idx);
			result->payload_output_columns.push_back(column_idx);
		}
		column_idx++;
	}
	if (!key_types.empty()) {
		result->key_modifiers = gstate.planner->Modifiers();
		result->decoded_keys.Initialize(Allocator::DefaultAllocator(), key_types);
	}
	return std::move(result);
}

static void DecodeNextUnit(ClientContext &context, const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                           SortedMergeLocalState &lstate, idx_t input_idx, SortedMergeSegmentReader &reader) {
	auto row_number_idx = lstate.payload_columns.size();
	if (!reader.scanner) {
		auto column_ids = lstate.payload_scan_columns;
		column_ids.push_back(MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER);
		auto filters = make_uniq<TableFilterSet>();
		ProjectionIndex row_number_column(row_number_idx);
		filters->PushFilter(row_number_column, ExpressionFilter::CreateComparisonFilter(
		                                           ExpressionType::COMPARE_GREATERTHANOREQUALTO,
		                                           Value::BIGINT(NumericCast<int64_t>(reader.next_unit_start))));
		filters->PushFilter(row_number_column,
		                    ExpressionFilter::CreateComparisonFilter(
		                        ExpressionType::COMPARE_LESSTHAN, Value::BIGINT(NumericCast<int64_t>(reader.row_end))));
		reader.scanner = make_uniq<DuckLakeSingleFileScanner>(context, bind_data.table, *gstate.inputs[input_idx],
		                                                      column_ids, std::move(filters));
		reader.scan_chunk.Initialize(context, reader.scanner->GetTypes());
	}
	auto unit = make_shared_ptr<SortedMergeDecodeUnit>();
	unit->row_start = reader.next_unit_start;
	unit->rows = make_uniq<ColumnDataCollection>(context, lstate.payload_types);
	DataChunk payload;
	payload.InitializeEmpty(lstate.payload_types);
	// scanned chunks are only valid until the next scan, so copy them
	while (unit->rows->SizeInBytes() < gstate.unit_size_bytes) {
		if (!reader.scanner->Scan(reader.scan_chunk)) {
			reader.exhausted = true;
			// the scanner holds the file's buffers, which the segment no longer needs
			reader.scanner.reset();
			break;
		}
		payload.ReferenceColumns(reader.scan_chunk, lstate.payload_columns);
		unit->rows->Append(payload);
		for (auto entry : reader.scan_chunk.data[row_number_idx].Values<int64_t>()) {
			unit->row_numbers.push_back(NumericCast<idx_t>(entry.GetValue()));
		}
		if (unit->row_numbers.back() + 1 == reader.row_end) {
			// the last row of the segment is decoded, release the scanner without another scan
			reader.exhausted = true;
			reader.scanner.reset();
			break;
		}
	}
	if (unit->row_numbers.empty()) {
		return;
	}
	unit->row_end = reader.exhausted ? reader.row_end : unit->row_numbers.back() + 1;
	reader.next_unit_start = unit->row_end;
	reader.units.push_back(std::move(unit));
}

//! Take references to the units covering the cursor's rows, decoding the input up to them if needed
static void TakeUnits(ClientContext &context, const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                      SortedMergeLocalState &lstate, idx_t input_idx, SortedMergeInputCursor &cursor) {
	auto &input_reader = *gstate.readers[input_idx];
	auto segment_size = gstate.segment_size;
	auto first_segment = cursor.row_start / segment_size;
	auto last_segment = (cursor.row_end - 1) / segment_size;
	vector<reference<SortedMergeSegmentReader>> segment_readers;
	{
		lock_guard<mutex> guard(input_reader.lock);
		for (auto segment_idx = first_segment; segment_idx <= last_segment; segment_idx++) {
			auto &entry = input_reader.segments[segment_idx];
			if (!entry) {
				entry = make_shared_ptr<SortedMergeSegmentReader>(segment_idx * segment_size,
				                                                  (segment_idx + 1) * segment_size);
			}
			segment_readers.push_back(*entry);
		}
	}
	auto take_segment_units = [&](SortedMergeSegmentReader &reader, vector<shared_ptr<SortedMergeDecodeUnit>> &units) {
		auto needed_end = MinValue(cursor.row_end, reader.row_end);
		while (!reader.exhausted && (reader.units.empty() || reader.units.back()->row_end < needed_end)) {
			DecodeNextUnit(context, bind_data, gstate, lstate, input_idx, reader);
		}
		for (auto &unit : reader.units) {
			if (unit->row_end > cursor.row_start && unit->row_start < cursor.row_end) {
				units.push_back(unit);
			}
		}
	};
	// decode the segments no other worker is decoding first, then wait for the rest
	vector<vector<shared_ptr<SortedMergeDecodeUnit>>> segment_units(segment_readers.size());
	vector<bool> taken(segment_readers.size(), false);
	for (idx_t i = 0; i < segment_readers.size(); i++) {
		auto &reader = segment_readers[i].get();
		unique_lock<mutex> guard(reader.lock, std::try_to_lock);
		if (guard.owns_lock()) {
			take_segment_units(reader, segment_units[i]);
			taken[i] = true;
		}
	}
	for (idx_t i = 0; i < segment_readers.size(); i++) {
		if (!taken[i]) {
			auto &reader = segment_readers[i].get();
			lock_guard<mutex> guard(reader.lock);
			take_segment_units(reader, segment_units[i]);
		}
	}
	for (auto &units : segment_units) {
		for (auto &unit : units) {
			cursor.units.push_back(std::move(unit));
		}
	}

	lock_guard<mutex> guard(input_reader.lock);
	input_reader.pending_batch_starts.erase(input_reader.pending_batch_starts.find(cursor.row_start));
	// drop the segments and units no batch can still need
	auto needed_row = input_reader.planned_row;
	if (!input_reader.pending_batch_starts.empty()) {
		needed_row = MinValue(needed_row, *input_reader.pending_batch_starts.begin());
	}
	auto &segments = input_reader.segments;
	while (!segments.empty() && segments.begin()->second->row_end <= needed_row) {
		segments.erase(segments.begin());
	}
	if (!segments.empty()) {
		auto &reader = *segments.begin()->second;
		lock_guard<mutex> reader_guard(reader.lock);
		while (!reader.units.empty() && reader.units.front()->row_end <= needed_row) {
			reader.units.pop_front();
		}
	}
}

//! Decode the unit holding the next row the planner hands out of an input, if no worker has. Returns false when there
//! is no such unit to decode.
static bool DecodeAhead(ClientContext &context, const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                        SortedMergeLocalState &lstate) {
	if (lstate.payload_columns.empty()) {
		return false;
	}
	auto input_count = gstate.readers.size();
	auto start = gstate.decode_ahead_cursor++;
	for (idx_t i = 0; i < input_count; i++) {
		auto input_idx = (start + i) % input_count;
		auto &input_reader = *gstate.readers[input_idx];
		shared_ptr<SortedMergeSegmentReader> reader;
		idx_t next_row;
		{
			lock_guard<mutex> guard(input_reader.lock);
			next_row = input_reader.planned_row;
			if (next_row == NumericLimits<idx_t>::Maximum()) {
				continue;
			}
			auto segment_idx = next_row / gstate.segment_size;
			auto &entry = input_reader.segments[segment_idx];
			if (!entry) {
				entry = make_shared_ptr<SortedMergeSegmentReader>(segment_idx * gstate.segment_size,
				                                                  (segment_idx + 1) * gstate.segment_size);
			}
			reader = entry;
		}
		unique_lock<mutex> reader_guard(reader->lock, std::try_to_lock);
		if (!reader_guard.owns_lock() || reader->exhausted || reader->next_unit_start > next_row) {
			continue;
		}
		DecodeNextUnit(context, bind_data, gstate, lstate, input_idx, *reader);
		gstate.units_decoded_ahead++;
		return true;
	}
	return false;
}

static void WakeParkedWorkers(SortedMergeGlobalState &gstate) {
	annotated_lock_guard<annotated_mutex> guard(gstate.lock);
	gstate.UnblockTasks();
}

SortedMergeLocalState::~SortedMergeLocalState() {
	if (!has_batch) {
		return;
	}
	annotated_lock_guard<annotated_mutex> guard(gstate.lock);
	gstate.batches_in_flight--;
	gstate.UnblockTasks();
}

enum class SortedMergeBatchResult : uint8_t { LOADED, FINISHED, PARKED };

static SortedMergeBatchResult LoadSortedMergeBatch(ClientContext &context, const SortedMergeBindData &bind_data,
                                                   SortedMergeGlobalState &gstate, SortedMergeLocalState &lstate,
                                                   OperatorSourceInput &input) {
	// a worker over the limit loads sort keys and decodes units ahead for the next batches before it parks
	bool nothing_to_decode = false;
	while (true) {
		annotated_unique_lock<annotated_mutex> guard(gstate.lock);
		auto max_memory = BufferManager::GetBufferManager(context).GetMaxMemory();
		auto worker_limit = gstate.worker_policy->WorkerLimit(max_memory);
		if (worker_limit != gstate.worker_limit) {
			DUCKDB_LOG_DEBUG(context,
			                 StringUtil::Format("DuckLake sorted merge: %llu workers for batches of %llu bytes "
			                                    "under a memory limit of %llu bytes",
			                                    worker_limit, gstate.worker_policy->AverageBatchBytes(), max_memory));
			gstate.worker_limit = worker_limit;
			if (gstate.worker_policy->AverageBatchBytes() > 0) {
				gstate.min_worker_limit = MinValue(gstate.min_worker_limit, worker_limit);
				gstate.max_worker_limit = MaxValue(gstate.max_worker_limit, worker_limit);
			}
		}
		if (gstate.planner_exhausted) {
			return SortedMergeBatchResult::FINISHED;
		}
		// when the pipeline no longer lets tasks park (it is finishing), the worker goes ahead
		if (gstate.batches_in_flight >= worker_limit && input.interrupt_state.CanCallback()) {
			// park at the batch boundary. First report the next batch index the planner hands out, so the sink takes
			// the finished batch and moves on: every batch below it is done or being emitted by a running worker. The
			// pipeline calls back right after taking the new index, and that call parks.
			auto floor = gstate.next_batch_index.load();
			if (gstate.requires_batch_index && floor > lstate.batch_index) {
				lstate.batch_index = floor;
				input.batch_index_state = SourceBatchIndexState::ADVANCED;
				return SortedMergeBatchResult::PARKED;
			}
			if (!nothing_to_decode) {
				guard.unlock();
				nothing_to_decode = !gstate.planner->Prefetch() && !DecodeAhead(context, bind_data, gstate, lstate);
				continue;
			}
			if (gstate.BlockTask(input.interrupt_state)) {
				gstate.parks++;
				return SortedMergeBatchResult::PARKED;
			}
		}
		gstate.batches_in_flight++;
		gstate.peak_batches_in_flight = MaxValue(gstate.peak_batches_in_flight, gstate.batches_in_flight);
		break;
	}
	{
		unique_lock<mutex> guard(gstate.plan_lock, std::try_to_lock);
		while (!guard.owns_lock()) {
			// load sort keys ahead for the worker that is planning. A worker still reporting its finished batch
			// would hold back the sink meanwhile, so it first reports the next batch index, like a parked worker.
			auto floor = gstate.next_batch_index.load();
			if (gstate.requires_batch_index && floor > lstate.batch_index) {
				{
					annotated_lock_guard<annotated_mutex> admission_guard(gstate.lock);
					gstate.batches_in_flight--;
				}
				WakeParkedWorkers(gstate);
				lstate.batch_index = floor;
				input.batch_index_state = SourceBatchIndexState::ADVANCED;
				return SortedMergeBatchResult::PARKED;
			}
			if (!gstate.planner->Prefetch()) {
				guard.lock();
				break;
			}
			guard.try_lock();
		}
		if (!lstate.key_output_columns.empty()) {
			lstate.batch_keys = make_uniq<DuckLakeMergeKeys>();
			lstate.batch_keys->keys.reserve(bind_data.row_group_size);
			lstate.key_offset = 0;
		}
		if (!gstate.planner->NextBatch(bind_data.row_group_size, lstate.slices, lstate.batch_keys.get())) {
			{
				annotated_lock_guard<annotated_mutex> admission_guard(gstate.lock);
				gstate.batches_in_flight--;
				gstate.planner_exhausted = true;
			}
			WakeParkedWorkers(gstate);
			return SortedMergeBatchResult::FINISHED;
		}
		lstate.batch_index = gstate.next_batch_index++;
		// the slices of one input in a batch cover a contiguous range of its live rows
		for (auto &slice : lstate.slices) {
			auto entry = lstate.cursors.find(slice.input_idx);
			if (entry == lstate.cursors.end()) {
				auto &cursor = lstate.cursors[slice.input_idx];
				cursor.row_start = slice.row_start;
				cursor.row_end = slice.row_end;
				continue;
			}
			entry->second.row_end = slice.row_end;
		}
		if (lstate.payload_columns.empty()) {
			// every column is decoded from the sort keys, nothing is read from the files
			lstate.cursors.clear();
		}
		for (auto &entry : lstate.cursors) {
			auto &reader = *gstate.readers[entry.first];
			lock_guard<mutex> reader_guard(reader.lock);
			reader.pending_batch_starts.insert(entry.second.row_start);
			auto next_row = gstate.planner->NextRow(entry.first);
			reader.planned_row = next_row.IsValid() ? next_row.GetIndex() : NumericLimits<idx_t>::Maximum();
		}
	}
	// parked workers report a higher batch index
	WakeParkedWorkers(gstate);
	for (auto &entry : lstate.cursors) {
		TakeUnits(context, bind_data, gstate, lstate, entry.first, entry.second);
	}
	lstate.slice_idx = 0;
	lstate.slice_offset = 0;
	lstate.batch_bytes = 0;
	lstate.has_batch = true;
	return SortedMergeBatchResult::LOADED;
}

static void FinishSortedMergeBatch(ExecutionContext &context, const PhysicalOperator &op,
                                   const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                                   SortedMergeLocalState &lstate) {
	idx_t batch_rows = 0;
	for (auto &slice : lstate.slices) {
		batch_rows += slice.row_count;
	}
	// release the decode units and sort keys of the batch
	lstate.cursors.clear();
	lstate.batch_keys.reset();
	lstate.has_batch = false;
	{
		annotated_lock_guard<annotated_mutex> guard(gstate.lock);
		gstate.batches_in_flight--;
		gstate.batches_emitted++;
		if (batch_rows >= bind_data.row_group_size) {
			gstate.worker_policy->AddBatch(lstate.batch_bytes,
			                               BufferManager::GetBufferManager(context.client).GetMaxMemory());
		}
		if (QueryProfiler::Get(context.client).IsEnabled()) {
			// the profiler keeps the extra info of the thread that flushes last, so write the merge-wide numbers
			InsertionOrderPreservingMap<string> info;
			info["Worker Limit"] = to_string(gstate.worker_limit);
			if (gstate.max_worker_limit > 0) {
				info["Worker Limit Range"] =
				    to_string(gstate.min_worker_limit) + " - " + to_string(gstate.max_worker_limit);
			}
			info["Peak Active Workers"] = to_string(gstate.peak_batches_in_flight);
			info["Average Batch Bytes"] = to_string(gstate.worker_policy->AverageBatchBytes());
			info["Batches"] = to_string(gstate.batches_emitted);
			info["Parks"] = to_string(gstate.parks);
			info["Units Decoded Ahead"] = to_string(gstate.units_decoded_ahead.load());
			info["Columns From Sort Keys"] = to_string(gstate.planner->DecodableColumns().size());
			info["Decode Unit Bytes"] = to_string(gstate.unit_size_bytes);
			auto &metrics = context.thread.profiler.GetOperatorMetrics(op);
			metrics.name = op.GetName();
			metrics.SetExtraInfo(std::move(info));
		}
	}
	WakeParkedWorkers(gstate);
}

//! The bytes the sink's collection takes for these rows: the fixed-size values and the strings stored out of line
static idx_t SinkBytes(const DataChunk &chunk, idx_t start) {
	idx_t result = 0;
	for (auto &vector : chunk.data) {
		auto physical_type = vector.GetType().InternalType();
		result += GetTypeIdSize(physical_type) * (chunk.size() - start);
		if (physical_type != PhysicalType::VARCHAR) {
			continue;
		}
		auto values = vector.Values<string_t>();
		for (idx_t row = start; row < chunk.size(); row++) {
			auto entry = values[row];
			if (entry.IsValid() && !entry.GetValueUnsafe().IsInlined()) {
				result += entry.GetValueUnsafe().GetSize();
			}
		}
	}
	return result;
}

//! Position the cursor at the next row to read, returns the rows available in its current chunk
static idx_t PositionCursor(SortedMergeInputCursor &cursor, const DuckLakeMergeSlice &slice) {
	if (!cursor.positioned) {
		// the first slice of the input in this batch starts inside its first unit
		auto &unit = *cursor.units[0];
		auto entry = std::lower_bound(unit.row_numbers.begin(), unit.row_numbers.end(), slice.row_start);
		cursor.unit_row = NumericCast<idx_t>(entry - unit.row_numbers.begin());
		unit.rows->InitializeScan(cursor.scan_state);
		cursor.chunk.Initialize(Allocator::DefaultAllocator(), unit.rows->Types());
		cursor.positioned = true;
	}
	while (cursor.unit_row == cursor.units[cursor.unit_idx]->rows->Count()) {
		cursor.unit_idx++;
		cursor.unit_row = 0;
		cursor.units[cursor.unit_idx]->rows->InitializeScan(cursor.scan_state);
	}
	auto &rows = *cursor.units[cursor.unit_idx]->rows;
	if (!rows.Seek(cursor.unit_row, cursor.scan_state, cursor.chunk)) {
		throw InternalException("Sorted merge failed to seek to row %llu of a decoded unit", cursor.unit_row);
	}
	return cursor.scan_state.next_row_index - cursor.unit_row;
}

//! Decode the sort key columns of the next count rows of the batch into the output from row output_offset
static void DecodeKeyColumns(SortedMergeLocalState &lstate, DataChunk &output, idx_t output_offset, idx_t count) {
	auto &keys = lstate.batch_keys->keys;
	lstate.decoded_keys.Reset();
	for (idx_t i = 0; i < count; i++) {
		CreateSortKeyHelpers::DecodeSortKey(keys[lstate.key_offset + i], lstate.decoded_keys, i, lstate.key_modifiers);
	}
	lstate.decoded_keys.SetCardinality(count);
	for (idx_t c = 0; c < lstate.key_output_columns.size(); c++) {
		VectorOperations::Copy(lstate.decoded_keys.data[c], output.data[lstate.key_output_columns[c]], count, 0,
		                       output_offset);
	}
	lstate.key_offset += count;
}

static void EmitSortedMergeRows(SortedMergeLocalState &lstate, DataChunk &output) {
	auto start = output.size();
	auto decode_keys = !lstate.key_output_columns.empty();
	auto output_count = start;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	while (output_count < STANDARD_VECTOR_SIZE && lstate.slice_idx < lstate.slices.size()) {
		auto &slice = lstate.slices[lstate.slice_idx];
		auto count = MinValue(slice.row_count - lstate.slice_offset, STANDARD_VECTOR_SIZE - output_count);
		if (!lstate.payload_columns.empty()) {
			auto &cursor = lstate.cursors[slice.input_idx];
			auto available = PositionCursor(cursor, slice);
			if (lstate.slice_offset == 0 &&
			    cursor.units[cursor.unit_idx]->row_numbers[cursor.unit_row] != slice.row_start) {
				throw InternalException("Sorted merge expected file row %llu of input %llu but found %llu",
				                        slice.row_start, slice.input_idx,
				                        cursor.units[cursor.unit_idx]->row_numbers[cursor.unit_row]);
			}
			count = MinValue(count, available);
			auto chunk_offset = cursor.unit_row - cursor.scan_state.current_row_index;
			for (idx_t i = 0; i < count; i++) {
				sel.set_index(i, chunk_offset + i);
			}
			if (decode_keys) {
				for (idx_t c = 0; c < lstate.payload_columns.size(); c++) {
					VectorOperations::Copy(cursor.chunk.data[c], output.data[lstate.payload_output_columns[c]], sel,
					                       count, 0, output_count);
				}
			} else {
				output.Append(cursor.chunk, sel, count);
			}
			cursor.unit_row += count;
		}
		output_count += count;

		lstate.slice_offset += count;
		if (lstate.slice_offset == slice.row_count) {
			lstate.slice_idx++;
			lstate.slice_offset = 0;
		}
	}
	if (decode_keys) {
		// the sort keys are in merged order, so the key columns of every slice decode at once
		DecodeKeyColumns(lstate, output, start, output_count - start);
		output.SetCardinality(output_count);
	}
	lstate.batch_bytes += SinkBytes(output, start);
}

//===--------------------------------------------------------------------===//
// Sorted Merge Source
//===--------------------------------------------------------------------===//
class DuckLakeSortedMergeSource : public PhysicalOperator {
public:
	DuckLakeSortedMergeSource(PhysicalPlan &physical_plan, const vector<LogicalType> &types,
	                          shared_ptr<SortedMergeBindData> bind_data_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types, 0),
	      bind_data(std::move(bind_data_p)) {
	}

	shared_ptr<SortedMergeBindData> bind_data;

public:
	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

	bool SupportsPartitioning(const OperatorPartitionInfo &partition_info) const override {
		return !partition_info.RequiresPartitionColumns();
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context,
	                                                   const OperatorPartitionInfo &partition_info) const override {
		auto result = InitSortedMergeGlobal(context, *bind_data);
		result->requires_batch_index = partition_info.RequiresBatchIndex();
		return std::move(result);
	}

	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override {
		return InitSortedMergeLocal(*bind_data, gstate.Cast<SortedMergeGlobalState>());
	}

	OperatorPartitionData GetPartitionData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
	                                       LocalSourceState &lstate,
	                                       const OperatorPartitionInfo &partition_info) const override {
		return OperatorPartitionData(lstate.Cast<SortedMergeLocalState>().batch_index);
	}

	string GetName() const override {
		return "DUCKLAKE_SORTED_MERGE";
	}

protected:
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		auto &gstate = input.global_state.Cast<SortedMergeGlobalState>();
		auto &lstate = input.local_state.Cast<SortedMergeLocalState>();
		while (chunk.size() == 0) {
			if (!lstate.has_batch) {
				switch (LoadSortedMergeBatch(context.client, *bind_data, gstate, lstate, input)) {
				case SortedMergeBatchResult::FINISHED:
					return SourceResultType::FINISHED;
				case SortedMergeBatchResult::PARKED:
					return SourceResultType::BLOCKED;
				case SortedMergeBatchResult::LOADED:
					break;
				}
			}
			EmitSortedMergeRows(lstate, chunk);
			if (lstate.slice_idx == lstate.slices.size()) {
				FinishSortedMergeBatch(context, *this, *bind_data, gstate, lstate);
			}
		}
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
};

class DuckLakeLogicalSortedMerge : public LogicalExtensionOperator {
public:
	DuckLakeLogicalSortedMerge(TableIndex table_index, shared_ptr<SortedMergeBindData> bind_data_p,
	                           vector<LogicalType> result_types)
	    : table_index(table_index), bind_data(std::move(bind_data_p)), result_types(std::move(result_types)) {
	}

	TableIndex table_index;
	shared_ptr<SortedMergeBindData> bind_data;
	vector<LogicalType> result_types;

public:
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		return planner.Make<DuckLakeSortedMergeSource>(types, bind_data);
	}

	string GetName() const override {
		return "DUCKLAKE_SORTED_MERGE";
	}

	string GetExtensionName() const override {
		return "ducklake";
	}

	vector<ColumnBinding> GetColumnBindings() override {
		return GenerateColumnBindings(table_index, result_types.size());
	}

	void ResolveTypes() override {
		types = result_types;
	}
};

static unique_ptr<LogicalOperator> DuckLakeSortedMergeBind(ClientContext &context, TableFunctionBindInput &input,
                                                           TableIndex bind_index, vector<Identifier> &return_names) {
	vector<DuckLakeFileListEntry> files;
	idx_t row_group_size;
	auto &table = BindSortedMergeTable(context, input, files, row_group_size);
	vector<LogicalType> return_types;
	for (auto &column : table.GetColumns().Physical()) {
		return_names.push_back(column.GetName());
		return_types.push_back(column.Type());
	}
	auto bind_data = make_shared_ptr<SortedMergeBindData>(table);
	bind_data->files = std::move(files);
	bind_data->row_group_size = row_group_size;
	Value decode_unit_size;
	if (DuckLakeTableFunctionUtil::TryGetNonNullOption(input, "decode_unit_size", "integer", decode_unit_size)) {
		bind_data->decode_unit_size = NumericCast<idx_t>(MaxValue<int64_t>(decode_unit_size.GetValue<int64_t>(), 1));
	}
	return make_uniq<DuckLakeLogicalSortedMerge>(bind_index, std::move(bind_data), std::move(return_types));
}

//===--------------------------------------------------------------------===//
// ducklake_sorted_merge_worker_limits
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> DuckLakeSortedMergeWorkerLimitsBind(ClientContext &context,
                                                                    TableFunctionBindInput &input,
                                                                    vector<LogicalType> &return_types,
                                                                    vector<Identifier> &names) {
	for (auto &value : input.inputs) {
		if (value.IsNull()) {
			throw BinderException("ducklake_sorted_merge_worker_limits does not accept NULL arguments");
		}
	}
	auto max_memory = DBConfig::ParseMemoryLimit(StringValue::Get(input.inputs[0]));
	auto threads = input.inputs[1].GetValue<int64_t>();
	if (threads <= 0) {
		throw BinderException("threads must be positive");
	}
	names.emplace_back("batch");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("batch_bytes");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("average_batch_bytes");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("worker_limit");
	return_types.emplace_back(LogicalType::BIGINT);

	DuckLakeSortedMergeWorkerPolicy policy(NumericCast<idx_t>(threads));
	auto result = make_uniq<MetadataBindData>();
	idx_t batch = 0;
	for (auto &entry : ListValue::GetChildren(input.inputs[2])) {
		auto batch_bytes = entry.GetValue<int64_t>();
		if (batch_bytes < 0) {
			throw BinderException("batch sizes must not be negative");
		}
		policy.AddBatch(NumericCast<idx_t>(batch_bytes), max_memory);
		vector<Value> row_values;
		row_values.push_back(Value::BIGINT(NumericCast<int64_t>(batch++)));
		row_values.push_back(Value::BIGINT(batch_bytes));
		row_values.push_back(Value::BIGINT(NumericCast<int64_t>(policy.AverageBatchBytes())));
		row_values.push_back(Value::BIGINT(NumericCast<int64_t>(policy.WorkerLimit(max_memory))));
		result->rows.push_back(std::move(row_values));
	}
	return std::move(result);
}

static void DuckLakeSortedMergeWorkerLimitsExecute(ClientContext &context, TableFunctionInput &data,
                                                   DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<MetadataBindData>();
	auto &state = data.global_state->Cast<DuckLakeRunOnceState>();
	DuckLakeBaseMetadataFunction::ScanRows(bind_data.rows, state.offset, output);
}

DuckLakeSortedMergeWorkerLimitsFunction::DuckLakeSortedMergeWorkerLimitsFunction()
    : TableFunction("ducklake_sorted_merge_worker_limits",
                    FunctionSignature()
                        .AddPositionalOnly("memory_limit", LogicalType::VARCHAR)
                        .AddPositionalOnly("threads", LogicalType::BIGINT)
                        .AddPositionalOnly("batch_bytes", LogicalType::LIST(LogicalType::BIGINT)),
                    DuckLakeSortedMergeWorkerLimitsExecute, DuckLakeSortedMergeWorkerLimitsBind,
                    DuckLakeRunOnceState::Init) {
}

DuckLakeSortedMergeFunction::DuckLakeSortedMergeFunction()
    : TableFunction("ducklake_sorted_merge", FunctionSignature().AddPositionalOnly("catalog", LogicalType::VARCHAR),
                    nullptr, nullptr, nullptr) {
	AddSortedMergeArguments(GetSignature());
	bind_operator = DuckLakeSortedMergeBind;
}

} // namespace duckdb
