#include "functions/ducklake_table_functions.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/function/scalar_function.hpp"
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
#include "duckdb/storage/temporary_memory_manager.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "storage/ducklake_catalog.hpp"
#include "storage/ducklake_multi_file_reader.hpp"
#include "storage/ducklake_multisequence_select.hpp"
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
	if (!DuckLakeSortedMergeSequencer::HasMergeableSort(table)) {
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
// ducklake_sorted_merge_sequence
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> DuckLakeSortedMergeSequenceBind(ClientContext &context, TableFunctionBindInput &input,
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
	DuckLakeSortedMergeSequencer sequencer(context, table, inputs, row_group_size * ROW_GROUPS_PER_KEY_SEGMENT);
	auto result = make_uniq<MetadataBindData>();
	auto &allocator = Allocator::DefaultAllocator();
	auto buffers = DuckLakeSortedMergeSequencer::CreateOrderBuffers(allocator);
	DuckLakeAllocatedArray<DuckLakeMergeRun> runs(allocator);
	vector<DuckLakeMergeSlice> slices;
	DuckLakeSequencedBatch batch;
	for (idx_t row_group = 0; sequencer.NextBatch(row_group_size, batch); row_group++) {
		DuckLakeSortedMergeSequencer::OrderBatch(batch, *buffers, runs);
		DuckLakeSortedMergeSequencer::RunSlices(batch, runs, slices);
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

DuckLakeSortedMergeSequenceFunction::DuckLakeSortedMergeSequenceFunction()
    : DuckLakeBaseMetadataFunction("ducklake_sorted_merge_sequence", DuckLakeSortedMergeSequenceBind) {
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
	//! Whether each row's snapshot id follows the table's columns, as compaction writes it
	bool write_snapshot_id = false;
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

	~SortedMergeSegmentReader() {
		CloseScanner();
	}

	mutex lock;
	//! One past the last file row of the segment
	idx_t row_end;
	//! The scanner, while the segment is being decoded, and the count of open scanners it is in
	unique_ptr<DuckLakeSingleFileScanner> scanner;
	optional_ptr<atomic<idx_t>> open_scanners;
	DataChunk scan_chunk;
	bool exhausted = false;
	//! The first file row of the next unit
	idx_t next_unit_start;
	//! Decoded units that a batch may still need
	deque<shared_ptr<SortedMergeDecodeUnit>> units;

	void OpenScanner(unique_ptr<DuckLakeSingleFileScanner> scanner_p, atomic<idx_t> &open_scanners_p) {
		scanner = std::move(scanner_p);
		open_scanners = &open_scanners_p;
		(*open_scanners)++;
	}
	//! The scanner holds the file's buffers, which the segment no longer needs once it is decoded
	void CloseScanner() {
		if (scanner) {
			scanner.reset();
			(*open_scanners)--;
		}
	}
};

//! The segments of one input, each read by its own reader so different parts of a file decode in parallel
struct SortedMergeInputReader {
	mutex lock;
	//! Segment readers by segment index
	map<idx_t, shared_ptr<SortedMergeSegmentReader>> segments;
	//! The first row of each admitted batch that has not yet taken its units
	multiset<idx_t> pending_batch_starts;
	//! Rows before this are sequenced, so no batch that is not yet admitted needs them
	idx_t sequenced_row = 0;
};

//! The inherited lock guards the batch counts and the parked workers
struct SortedMergeGlobalState : public GlobalSourceState {
	~SortedMergeGlobalState() override {
		// every segment closes its scanner
		readers.clear();
		D_ASSERT(open_payload_scanners == 0);
	}

	//! Segments of payload with an open scanner, and how many the memory reservation allows decoding ahead to keep open
	atomic<idx_t> open_payload_scanners {0};
	atomic<idx_t> max_payload_scanners_ahead {0};
	//! Guards the sequencer
	mutex sequence_lock;
	vector<unique_ptr<DuckLakeFileScanBind>> inputs;
	unique_ptr<DuckLakeSortedMergeSequencer> sequencer;
	vector<unique_ptr<SortedMergeInputReader>> readers;
	//! The batch index of the next batch the sequencer hands out
	atomic<idx_t> next_batch_index {0};
	idx_t max_threads = 1;
	//! Whether the sink orders rows by batch index, so a parked worker reports the batches it has finished
	bool requires_batch_index = false;
	//! Batches being emitted
	idx_t batches_in_flight = 0;
	//! Workers allowed from the decoded size of the batches, as the sink measures it
	unique_ptr<DuckLakeSortedMergeWorkerPolicy> worker_policy;
	//! The memory of the merge itself, reserved once the first batch is measured: the decode units and key scanners
	//! of the merge, and the buffers and payload scanner of each worker
	unique_ptr<TemporaryMemoryState> memory_state;
	idx_t reserved_fixed_bytes = 0;
	idx_t reserved_worker_bytes = 0;
	//! The bytes per worker for ordering and emitting a batch
	idx_t worker_bytes = 0;
	//! The compressed bytes of a row group of the inputs, and the share of them the sort key columns take
	idx_t compressed_row_group_bytes = 0;
	//! The share of the decoded bytes of the batches the sort key columns take
	idx_t decoded_bytes = 0;
	idx_t decoded_key_bytes = 0;
	double key_column_share = 0;
	//! Whether columns other than the sort keys are read from the files
	bool has_payload = false;
	bool sequencer_exhausted = false;
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
	//! Whether workers waiting to start a batch decode units ahead
	bool decode_ahead = true;
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
	//! Whether the current batch reads the input
	bool active = false;

	void Reset() {
		units.clear();
		unit_idx = 0;
		unit_row = 0;
		chunk.Destroy();
		positioned = false;
		active = false;
	}
};

struct SortedMergeLocalState : public LocalSourceState {
	SortedMergeLocalState(SortedMergeGlobalState &gstate, Allocator &allocator)
	    : gstate(gstate), runs(allocator), order_buffers(DuckLakeSortedMergeSequencer::CreateOrderBuffers(allocator)) {
	}
	//! A worker that stops inside a batch, e.g. when a LIMIT is reached, gives its slot back to the parked workers
	~SortedMergeLocalState() override;

	SortedMergeGlobalState &gstate;
	idx_t batch_index = 0;
	bool has_batch = false;
	//! Decoded bytes of the rows emitted for the current batch
	idx_t batch_bytes = 0;
	//! The part of batch_bytes the sort key columns take
	idx_t batch_key_bytes = 0;
	//! The rows the sequencer chose for the batch, the rows of each input, and the rows in merged order
	DuckLakeSequencedBatch sequenced;
	vector<DuckLakeMergeSlice> input_rows;
	DuckLakeAllocatedArray<DuckLakeMergeRun> runs;
	idx_t run_idx = 0;
	idx_t run_offset = 0;
	//! The buffers the batch is ordered in, kept for the next batch
	shared_ptr<DuckLakeOrderBuffers> order_buffers;
	//! A cursor per input, and the inputs the current batch reads in the order it first reads them
	vector<unique_ptr<SortedMergeInputCursor>> cursors;
	vector<idx_t> batch_inputs;

	//! A parked worker gives back the memory of its buffers
	void ReleaseBuffers() {
		runs.Release();
		DuckLakeSortedMergeSequencer::ReleaseOrderBuffers(*order_buffers);
		if (batch_keys) {
			batch_keys->Clear();
			batch_keys->keys.Release();
		}
	}

	void ResetCursors() {
		for (auto input_idx : batch_inputs) {
			cursors[input_idx]->Reset();
		}
		batch_inputs.clear();
	}
	//! The rows of an output chunk gathered per input from the input's current chunk, as (row in that chunk, row of
	//! the output), copied into staging together before the input's chunk changes
	vector<vector<pair<sel_t, sel_t>>> pending_rows;
	vector<idx_t> pending_inputs;
	DataChunk staging;
	SelectionVector gather_sel {STANDARD_VECTOR_SIZE};
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
//! Memory limit per segment of sort keys the sequencer may load ahead
static constexpr idx_t KEY_SEGMENT_MEMORY = 64ULL << 20;

static unique_ptr<SortedMergeGlobalState> InitSortedMergeGlobal(ClientContext &context,
                                                                const SortedMergeBindData &bind_data) {
	auto result = make_uniq<SortedMergeGlobalState>();
	result->inputs = BindMergeInputs(context, bind_data.table, bind_data.files);
	// every input keeps about one unit decoded, so size units to keep that a small part of the memory limit
	auto max_memory = BufferManager::GetBufferManager(context).GetMaxMemory();
	result->sequencer = make_uniq<DuckLakeSortedMergeSequencer>(context, bind_data.table, result->inputs,
	                                                            bind_data.row_group_size * ROW_GROUPS_PER_KEY_SEGMENT);
	result->sequencer->Initialize();
	result->sequencer->SetBatchRows(bind_data.row_group_size);
	// load key segments ahead only when the memory limit leaves room for two per input, which takes more memory when
	// the merge also decodes other columns
	auto input_count = result->inputs.size();
	auto keys_only = result->sequencer->DecodableColumns().size() == bind_data.table.GetColumns().PhysicalColumnCount();
	auto segments_ahead = max_memory / (keys_only ? KEY_SEGMENT_MEMORY / 2 : KEY_SEGMENT_MEMORY);

	result->has_payload = !keys_only || bind_data.write_snapshot_id;
	result->sequencer->SetMaxSegmentsAhead(segments_ahead < 2 * input_count ? input_count : segments_ahead);
	idx_t file_bytes = 0;
	idx_t file_rows = 0;
	for (auto &input : result->inputs) {
		if (input->row_count.IsValid()) {
			file_bytes += input->file.file.file_size_bytes;
			file_rows += input->row_count.GetIndex();
		}
	}
	if (file_rows > 0) {
		result->compressed_row_group_bytes = file_bytes * bind_data.row_group_size / file_rows;
	}
	result->key_column_share = double(MaxValue<idx_t>(result->sequencer->DecodableColumns().size(), 1)) /
	                           double(bind_data.table.GetColumns().PhysicalColumnCount());
	for (idx_t input_idx = 0; input_idx < result->inputs.size(); input_idx++) {
		result->readers.push_back(make_uniq<SortedMergeInputReader>());
	}
	auto unit_size = max_memory / (UNIT_MEMORY_FRACTION * MaxValue<idx_t>(result->inputs.size(), 1));
	result->unit_size_bytes = MaxValue(MIN_UNIT_SIZE_BYTES, MinValue(MAX_UNIT_SIZE_BYTES, unit_size));
	if (bind_data.decode_unit_size.IsValid()) {
		result->unit_size_bytes = bind_data.decode_unit_size.GetIndex();
	}
	// with too many inputs for units to fit the budget, decoding ahead would only add to the overrun
	result->decode_ahead = result->unit_size_bytes * input_count <= max_memory / UNIT_MEMORY_FRACTION;
	result->segment_size = bind_data.row_group_size;
	result->max_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	auto worker_bytes = DuckLakeSortedMergeSequencer::WorkerBytes(bind_data.row_group_size,
	                                                              !result->sequencer->DecodableColumns().empty());
	result->worker_bytes = worker_bytes;
	result->worker_policy = make_uniq<DuckLakeSortedMergeWorkerPolicy>(result->max_threads, worker_bytes);
	return std::move(result);
}

static unique_ptr<SortedMergeLocalState>
InitSortedMergeLocal(ClientContext &context, const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate) {
	// the buffers that grow with the batch count against the memory limit
	auto &allocator = BufferAllocator::Get(context);
	auto result = make_uniq<SortedMergeLocalState>(gstate, allocator);
	for (idx_t input_idx = 0; input_idx < gstate.inputs.size(); input_idx++) {
		result->cursors.push_back(make_uniq<SortedMergeInputCursor>());
	}
	result->pending_rows.resize(gstate.inputs.size());
	auto &key_columns = gstate.sequencer->DecodableColumns();
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
	if (bind_data.write_snapshot_id) {
		auto snapshot_id_column = DuckLakeMultiFileReader::COLUMN_IDENTIFIER_SNAPSHOT_ID;
		result->payload_columns.push_back(result->payload_types.size());
		result->payload_types.push_back(bind_data.table.GetVirtualColumns().at(snapshot_id_column).type);
		result->payload_scan_columns.push_back(snapshot_id_column);
		result->payload_output_columns.push_back(column_idx);
	}
	if (!result->payload_types.empty()) {
		result->staging.Initialize(Allocator::DefaultAllocator(), result->payload_types);
	}
	if (!key_types.empty()) {
		result->batch_keys = make_uniq<DuckLakeMergeKeys>(allocator);
		result->key_modifiers = gstate.sequencer->Modifiers();
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
		reader.OpenScanner(make_uniq<DuckLakeSingleFileScanner>(context, bind_data.table, *gstate.inputs[input_idx],
		                                                        column_ids, std::move(filters)),
		                   gstate.open_payload_scanners);
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
			reader.CloseScanner();
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
			reader.CloseScanner();
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
	auto needed_row = input_reader.sequenced_row;
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

//! Decode the unit holding the next row the sequencer hands out of an input, if no worker has. Returns false when there
//! is no such unit to decode.
static bool DecodeAhead(ClientContext &context, const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                        SortedMergeLocalState &lstate) {
	if (!gstate.decode_ahead || lstate.payload_columns.empty()) {
		return false;
	}
	auto input_count = gstate.readers.size();
	auto start = gstate.decode_ahead_cursor++;
	for (idx_t i = 0; i < input_count; i++) {
		auto input_idx = (start + i) % input_count;
		if (!gstate.sequencer->InputIsNear(input_idx)) {
			continue;
		}
		auto &input_reader = *gstate.readers[input_idx];
		shared_ptr<SortedMergeSegmentReader> reader;
		idx_t next_row;
		{
			lock_guard<mutex> guard(input_reader.lock);
			next_row = input_reader.sequenced_row;
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
		if (!reader->scanner && gstate.open_payload_scanners >= gstate.max_payload_scanners_ahead) {
			// another open scanner would take more memory than the reservation allows
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
		if (gstate.sequencer_exhausted) {
			return SortedMergeBatchResult::FINISHED;
		}
		// when the pipeline no longer lets tasks park (it is finishing), the worker goes ahead
		if (gstate.batches_in_flight >= worker_limit && input.interrupt_state.CanCallback()) {
			// park at the batch boundary. First report the next batch index the sequencer hands out, so the sink takes
			// the finished batch and moves on: every batch below it is done or being emitted by a running worker. The
			// pipeline calls back right after taking the new index, and that call parks. A worker without a batch gives
			// back the memory of its buffers meanwhile.
			lstate.ReleaseBuffers();
			auto floor = gstate.next_batch_index.load();
			if (gstate.requires_batch_index && floor > lstate.batch_index) {
				lstate.batch_index = floor;
				input.batch_index_state = SourceBatchIndexState::ADVANCED;
				return SortedMergeBatchResult::PARKED;
			}
			if (!nothing_to_decode) {
				guard.unlock();
				nothing_to_decode = !gstate.sequencer->Prefetch() && !DecodeAhead(context, bind_data, gstate, lstate);
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
		unique_lock<mutex> guard(gstate.sequence_lock, std::try_to_lock);
		while (!guard.owns_lock()) {
			// load sort keys ahead for the worker that is sequencing. A worker still reporting its finished batch
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
			auto generation = gstate.sequencer->WaitGeneration();
			if (!gstate.sequencer->Prefetch()) {
				if (!gstate.sequencer->MergingKeys()) {
					guard.lock();
					break;
				}
				// nothing to load yet, wait until there is, or the sequencer is free
				gstate.sequencer->WaitForWork(generation);
			}
			guard.try_lock();
		}
		if (lstate.batch_keys) {
			lstate.batch_keys->Clear();
			lstate.key_offset = 0;
		}
		if (!gstate.sequencer->NextBatch(bind_data.row_group_size, lstate.sequenced)) {
			{
				annotated_lock_guard<annotated_mutex> admission_guard(gstate.lock);
				gstate.batches_in_flight--;
				gstate.sequencer_exhausted = true;
			}
			WakeParkedWorkers(gstate);
			return SortedMergeBatchResult::FINISHED;
		}
		lstate.batch_index = gstate.next_batch_index++;
		// a batch reads a contiguous range of the live rows of each of its inputs
		DuckLakeSortedMergeSequencer::BatchInputRows(lstate.sequenced, lstate.input_rows);
		for (auto &rows : lstate.input_rows) {
			auto &cursor = *lstate.cursors[rows.input_idx];
			cursor.active = true;
			cursor.row_start = rows.row_start;
			cursor.row_end = rows.row_end;
			lstate.batch_inputs.push_back(rows.input_idx);
		}
		if (lstate.payload_columns.empty()) {
			// every column is decoded from the sort keys, nothing is read from the files
			lstate.ResetCursors();
		}
		for (auto input_idx : lstate.batch_inputs) {
			auto &reader = *gstate.readers[input_idx];
			lock_guard<mutex> reader_guard(reader.lock);
			reader.pending_batch_starts.insert(lstate.cursors[input_idx]->row_start);
			auto next_row = gstate.sequencer->NextRow(input_idx);
			reader.sequenced_row = next_row.IsValid() ? next_row.GetIndex() : NumericLimits<idx_t>::Maximum();
		}
	}
	// the workers waiting for the sequencer can take it, and parked workers report a higher batch index
	gstate.sequencer->NotifyWaiters(true);
	WakeParkedWorkers(gstate);
	// put the batch's rows in merged order
	DuckLakeSortedMergeSequencer::OrderBatch(lstate.sequenced, *lstate.order_buffers, lstate.runs,
	                                         lstate.batch_keys.get());
	lstate.sequenced.Clear();
	// take units in the order the batch first reads from each input
	if (!lstate.batch_inputs.empty()) {
		lstate.batch_inputs.clear();
		vector<bool> listed(lstate.cursors.size(), false);
		for (auto &run : lstate.runs) {
			if (!listed[run.input_idx]) {
				listed[run.input_idx] = true;
				lstate.batch_inputs.push_back(run.input_idx);
			}
		}
	}
	for (auto input_idx : lstate.batch_inputs) {
		TakeUnits(context, bind_data, gstate, lstate, input_idx, *lstate.cursors[input_idx]);
	}
	lstate.run_idx = 0;
	lstate.run_offset = 0;
	lstate.batch_bytes = 0;
	lstate.batch_key_bytes = 0;
	lstate.has_batch = true;
	return SortedMergeBatchResult::LOADED;
}

//! A scanner holds a row group of its columns as read and decompressed, and the file cache keeps the bytes read
static idx_t ScannerBytes(idx_t decoded_bytes, double compression_ratio) {
	return decoded_bytes + idx_t(2 * compression_ratio * double(decoded_bytes));
}

//! Reserve the merge's own memory with the temporary memory manager, and cap the workers to what it grants. Called at
//! the end of every batch, which also sees other operators change their reservations. The caller holds the lock of
//! the global state.
static void UpdateMergeReservation(ClientContext &context, SortedMergeGlobalState &gstate) {
	// the decoded bytes of a batch, and how much the inputs compress them, assuming the columns compress alike
	auto batch_bytes = gstate.worker_policy->AverageBatchBytes();
	auto compression_ratio =
	    gstate.compressed_row_group_bytes > 0 ? double(gstate.compressed_row_group_bytes) / double(batch_bytes) : 1.0;
	auto key_share = gstate.decoded_key_bytes > 0 ? double(gstate.decoded_key_bytes) / double(gstate.decoded_bytes)
	                                              : gstate.key_column_share;
	auto key_bytes = idx_t(key_share * double(batch_bytes));
	idx_t fixed_bytes = gstate.sequencer->MaxSegmentsLoaded() * ScannerBytes(key_bytes, compression_ratio);
	idx_t per_worker = gstate.worker_bytes;
	idx_t payload_scanner = 0;
	if (gstate.has_payload) {
		// a unit in use and one decoding for every input
		fixed_bytes += 2 * gstate.inputs.size() * gstate.unit_size_bytes;
		// a worker decodes one segment of payload at a time
		payload_scanner = ScannerBytes(batch_bytes - MinValue(key_bytes, batch_bytes), compression_ratio);
		per_worker += payload_scanner;
	}
	if (!gstate.memory_state) {
		gstate.memory_state = TemporaryMemoryManager::Get(context).Register(context);
	}
	if (fixed_bytes != gstate.reserved_fixed_bytes || per_worker != gstate.reserved_worker_bytes) {
		gstate.memory_state->SetMinimumReservation(fixed_bytes + per_worker);
		gstate.memory_state->SetRemainingSizeAndUpdateReservation(context,
		                                                          fixed_bytes + gstate.max_threads * per_worker);
		gstate.reserved_fixed_bytes = fixed_bytes;
		gstate.reserved_worker_bytes = per_worker;
		DUCKDB_LOG_DEBUG(context, StringUtil::Format("DuckLake sorted merge: reserved %llu bytes of %llu + %llu x %llu",
		                                             gstate.memory_state->GetReservation(), fixed_bytes,
		                                             gstate.max_threads, per_worker));
	} else {
		gstate.memory_state->UpdateReservation(context);
	}
	auto reservation = gstate.memory_state->GetReservation();
	gstate.worker_policy->SetReservation(fixed_bytes, per_worker, reservation);
	// decoding ahead may keep open the scanners of the workers, and of what the reservation has left over
	auto workers = gstate.worker_policy->WorkerLimit(BufferManager::GetBufferManager(context).GetMaxMemory());
	auto used = fixed_bytes + workers * per_worker;
	auto spare_scanners = payload_scanner > 0 && reservation > used ? (reservation - used) / payload_scanner : 0;
	gstate.max_payload_scanners_ahead = workers + spare_scanners;
}

static void FinishSortedMergeBatch(ExecutionContext &context, const PhysicalOperator &op,
                                   const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                                   SortedMergeLocalState &lstate) {
	idx_t batch_rows = 0;
	for (auto &run : lstate.runs) {
		batch_rows += run.row_count;
	}
	// release the decode units and sort keys of the batch
	lstate.ResetCursors();
	if (lstate.batch_keys) {
		lstate.batch_keys->Clear();
	}
	lstate.has_batch = false;
	{
		annotated_lock_guard<annotated_mutex> guard(gstate.lock);
		gstate.batches_in_flight--;
		gstate.batches_emitted++;
		if (batch_rows >= bind_data.row_group_size) {
			gstate.worker_policy->AddBatch(lstate.batch_bytes,
			                               BufferManager::GetBufferManager(context.client).GetMaxMemory());
			gstate.decoded_bytes += lstate.batch_bytes;
			gstate.decoded_key_bytes += lstate.batch_key_bytes;
		}
		if (gstate.worker_policy->AverageBatchBytes() > 0) {
			UpdateMergeReservation(context.client, gstate);
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
			info["Units Decoded Ahead"] =
			    gstate.decode_ahead ? to_string(gstate.units_decoded_ahead.load()) : "off (too many inputs)";
			info["Columns From Sort Keys"] = to_string(gstate.sequencer->DecodableColumns().size());
			info["Sections"] = to_string(gstate.sequencer->SectionCount());
			info["Sections Merged Without Keys"] = to_string(gstate.sequencer->KeylessSectionCount());
			info["Decode Unit Bytes"] = to_string(gstate.unit_size_bytes);
			if (gstate.memory_state) {
				info["Memory Reservation"] =
				    to_string(gstate.memory_state->GetReservation()) + " of " + to_string(gstate.reserved_fixed_bytes) +
				    " + " + to_string(gstate.max_threads) + " x " + to_string(gstate.reserved_worker_bytes);
			}
			auto &metrics = context.thread.profiler.GetOperatorMetrics(op);
			metrics.name = op.GetName();
			metrics.SetExtraInfo(std::move(info));
		}
	}
	WakeParkedWorkers(gstate);
}

//! The bytes the sink's collection takes for rows of a column: the fixed-size values and the strings stored out of line
static idx_t SinkBytes(const Vector &vector, idx_t start, idx_t end) {
	auto physical_type = vector.GetType().InternalType();
	idx_t result = GetTypeIdSize(physical_type) * (end - start);
	if (physical_type != PhysicalType::VARCHAR) {
		return result;
	}
	auto values = vector.Values<string_t>();
	for (idx_t row = start; row < end; row++) {
		auto entry = values[row];
		if (entry.IsValid() && !entry.GetValueUnsafe().IsInlined()) {
			result += entry.GetValueUnsafe().GetSize();
		}
	}
	return result;
}

static idx_t SinkBytes(const DataChunk &chunk, idx_t start) {
	idx_t result = 0;
	for (auto &vector : chunk.data) {
		result += SinkBytes(vector, start, chunk.size());
	}
	return result;
}

//! Position the cursor at the next row to read, returns the rows available in its current chunk
static idx_t PositionCursor(SortedMergeInputCursor &cursor, idx_t input_idx) {
	if (!cursor.positioned) {
		// the input's rows in this batch start inside its first unit
		auto &unit = *cursor.units[0];
		auto entry = std::lower_bound(unit.row_numbers.begin(), unit.row_numbers.end(), cursor.row_start);
		if (entry == unit.row_numbers.end() || *entry != cursor.row_start) {
			throw InternalException("Sorted merge expected file row %llu of input %llu in its first decoded unit",
			                        cursor.row_start, input_idx);
		}
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

//! Whether the cursor's chunk holds its next row, so positioning it keeps the chunk
static bool CursorChunkHoldsNextRow(const SortedMergeInputCursor &cursor) {
	return cursor.positioned && cursor.unit_row < cursor.units[cursor.unit_idx]->rows->Count() &&
	       cursor.scan_state.current_row_index <= cursor.unit_row && cursor.unit_row < cursor.scan_state.next_row_index;
}

//! Copy the rows gathered from an input's current chunk into staging, and point their output rows at them
static void StagePendingRows(SortedMergeLocalState &lstate, idx_t input_idx, SelectionVector &output_sel) {
	auto &rows = lstate.pending_rows[input_idx];
	if (rows.empty()) {
		return;
	}
	for (idx_t i = 0; i < rows.size(); i++) {
		lstate.gather_sel.set_index(i, rows[i].first);
	}
	auto base = lstate.staging.size();
	lstate.staging.Append(lstate.cursors[input_idx]->chunk, lstate.gather_sel, rows.size());
	for (idx_t i = 0; i < rows.size(); i++) {
		output_sel.set_index(rows[i].second, base + i);
	}
	rows.clear();
}

static void EmitSortedMergeRows(SortedMergeLocalState &lstate, DataChunk &output) {
	D_ASSERT(output.size() == 0);
	auto has_payload = !lstate.payload_columns.empty();
	// the rows of every input are copied once per chunk of the input, and the output selects them from staging
	SelectionVector output_sel(STANDARD_VECTOR_SIZE);
	if (has_payload) {
		lstate.staging.Reset();
	}
	idx_t output_count = 0;
	while (output_count < STANDARD_VECTOR_SIZE && lstate.run_idx < lstate.runs.size()) {
		auto &run = lstate.runs[lstate.run_idx];
		auto count = MinValue<idx_t>(run.row_count - lstate.run_offset, STANDARD_VECTOR_SIZE - output_count);
		if (has_payload) {
			auto &cursor = *lstate.cursors[run.input_idx];
			if (!CursorChunkHoldsNextRow(cursor)) {
				StagePendingRows(lstate, run.input_idx, output_sel);
			}
			auto available = PositionCursor(cursor, run.input_idx);
			count = MinValue(count, available);
			auto chunk_offset = cursor.unit_row - cursor.scan_state.current_row_index;
			auto &rows = lstate.pending_rows[run.input_idx];
			if (rows.empty()) {
				lstate.pending_inputs.push_back(run.input_idx);
			}
			for (idx_t i = 0; i < count; i++) {
				rows.emplace_back(UnsafeNumericCast<sel_t>(chunk_offset + i),
				                  UnsafeNumericCast<sel_t>(output_count + i));
			}
			cursor.unit_row += count;
		}
		output_count += count;

		lstate.run_offset += count;
		if (lstate.run_offset == run.row_count) {
			lstate.run_idx++;
			lstate.run_offset = 0;
		}
	}
	if (has_payload) {
		for (auto input_idx : lstate.pending_inputs) {
			StagePendingRows(lstate, input_idx, output_sel);
		}
		lstate.pending_inputs.clear();
		for (idx_t c = 0; c < lstate.payload_columns.size(); c++) {
			output.data[lstate.payload_output_columns[c]].Slice(lstate.staging.data[c], output_sel, output_count);
		}
	}
	if (!lstate.key_output_columns.empty()) {
		// the sort keys are in merged order, so the key columns of every run decode at once
		DecodeKeyColumns(lstate, output, 0, output_count);
	}
	output.SetCardinality(output_count);
	lstate.batch_bytes += SinkBytes(output, 0);
	for (auto column_idx : lstate.key_output_columns) {
		lstate.batch_key_bytes += SinkBytes(output.data[column_idx], 0, output_count);
	}
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
		return InitSortedMergeLocal(context.client, *bind_data, gstate.Cast<SortedMergeGlobalState>());
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
			if (lstate.run_idx == lstate.runs.size()) {
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

unique_ptr<LogicalOperator> PlanDuckLakeSortedMerge(DuckLakeTableEntry &table, vector<DuckLakeFileListEntry> files,
                                                    TableIndex table_index, idx_t row_group_size,
                                                    bool write_snapshot_id) {
	vector<LogicalType> return_types;
	for (auto &column : table.GetColumns().Physical()) {
		return_types.push_back(column.Type());
	}
	if (write_snapshot_id) {
		return_types.push_back(
		    table.GetVirtualColumns().at(DuckLakeMultiFileReader::COLUMN_IDENTIFIER_SNAPSHOT_ID).type);
	}
	auto bind_data = make_shared_ptr<SortedMergeBindData>(table);
	bind_data->files = std::move(files);
	bind_data->row_group_size = row_group_size;
	bind_data->write_snapshot_id = write_snapshot_id;
	return make_uniq<DuckLakeLogicalSortedMerge>(table_index, std::move(bind_data), std::move(return_types));
}

//===--------------------------------------------------------------------===//
// ducklake_multisequence_select
//===--------------------------------------------------------------------===//
//! Sorted lists of BIGINT keys, ordered by key and then by list
struct ListSequences {
	vector<vector<int64_t>> keys;

	idx_t Count() const {
		return keys.size();
	}
	bool Less(idx_t lhs_sequence, idx_t lhs_position, idx_t rhs_sequence, idx_t rhs_position) const {
		auto lhs = keys[lhs_sequence][lhs_position];
		auto rhs = keys[rhs_sequence][rhs_position];
		return lhs < rhs || (lhs == rhs && lhs_sequence < rhs_sequence);
	}
};

static void MultisequenceSelectFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &result_children = StructVector::GetEntries(result);
	auto &cuts_vector = result_children[0];
	auto steps = FlatVector::GetDataMutable<int64_t>(result_children[1]);
	for (idx_t row = 0; row < args.size(); row++) {
		auto sequences_value = args.GetValue(0, row);
		auto starts_value = args.GetValue(1, row);
		auto rank_value = args.GetValue(2, row);
		if (sequences_value.IsNull() || starts_value.IsNull() || rank_value.IsNull()) {
			throw InvalidInputException("ducklake_multisequence_select does not take NULL arguments");
		}
		ListSequences sequences;
		vector<idx_t> starts;
		vector<idx_t> ends;
		auto &starts_list = ListValue::GetChildren(starts_value);
		for (auto &sequence : ListValue::GetChildren(sequences_value)) {
			auto &sequence_keys = sequences.keys.emplace_back();
			for (auto &key : ListValue::GetChildren(sequence)) {
				sequence_keys.push_back(key.GetValue<int64_t>());
			}
			ends.push_back(sequence_keys.size());
		}
		if (starts_list.size() != ends.size()) {
			throw InvalidInputException("ducklake_multisequence_select needs a start for every sequence");
		}
		for (idx_t i = 0; i < ends.size(); i++) {
			auto start = starts_list[i].GetValue<int64_t>();
			if (start < 0 || NumericCast<idx_t>(start) > ends[i]) {
				throw InvalidInputException("ducklake_multisequence_select start %lld is out of range", start);
			}
			starts.push_back(NumericCast<idx_t>(start));
		}
		vector<idx_t> cuts;
		auto step_count = MultisequenceSelection<ListSequences>::Select(
		    sequences, starts, ends, NumericCast<idx_t>(rank_value.GetValue<int64_t>()), cuts);
		vector<Value> cut_values;
		for (auto cut : cuts) {
			cut_values.push_back(Value::BIGINT(NumericCast<int64_t>(cut)));
		}
		cuts_vector.SetValue(row, Value::LIST(LogicalType::BIGINT, std::move(cut_values)));
		steps[row] = NumericCast<int64_t>(step_count);
	}
	result.SetVectorType(args.AllConstant() ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
}

ScalarFunction DuckLakeMultisequenceSelectFunction() {
	child_list_t<LogicalType> result_children;
	result_children.emplace_back("cuts", LogicalType::LIST(LogicalType::BIGINT));
	result_children.emplace_back("steps", LogicalType::BIGINT);
	ScalarFunction function("ducklake_multisequence_select",
	                        {LogicalType::LIST(LogicalType::LIST(LogicalType::BIGINT)),
	                         LogicalType::LIST(LogicalType::BIGINT), LogicalType::BIGINT},
	                        LogicalType::STRUCT(std::move(result_children)), MultisequenceSelectFunction);
	function.SetFallible();
	return function;
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

	auto get_bytes = [&](const char *name, idx_t default_value, bool allow_zero = false) {
		Value value;
		if (!DuckLakeTableFunctionUtil::TryGetNonNullOption(input, name, "integer", value)) {
			return default_value;
		}
		auto bytes = value.GetValue<int64_t>();
		if (bytes < 0 || (bytes == 0 && !allow_zero)) {
			throw BinderException("%s must be %s", name, allow_zero ? "non-negative" : "positive");
		}
		return NumericCast<idx_t>(bytes);
	};
	auto worker_bytes =
	    get_bytes("worker_bytes", DuckLakeSortedMergeSequencer::WorkerBytes(DEFAULT_ROW_GROUP_SIZE, true));
	// the merge's memory reservation after each batch, for the fixed bytes of the merge and the bytes of each worker
	auto fixed_bytes = get_bytes("fixed_bytes", 0, true);
	auto per_worker_bytes = get_bytes("per_worker_bytes", worker_bytes);
	vector<Value> reservations;
	Value reservations_value;
	if (DuckLakeTableFunctionUtil::TryGetNonNullOption(input, "reservations", "list", reservations_value)) {
		reservations = ListValue::GetChildren(reservations_value);
	}
	auto &batches = ListValue::GetChildren(input.inputs[2]);
	if (!reservations.empty() && reservations.size() != batches.size()) {
		throw BinderException("reservations must have one entry per batch");
	}
	DuckLakeSortedMergeWorkerPolicy policy(NumericCast<idx_t>(threads), worker_bytes);
	auto result = make_uniq<MetadataBindData>();
	idx_t batch = 0;
	for (auto &entry : batches) {
		auto batch_bytes = entry.GetValue<int64_t>();
		if (batch_bytes < 0) {
			throw BinderException("batch sizes must not be negative");
		}
		policy.AddBatch(NumericCast<idx_t>(batch_bytes), max_memory);
		if (!reservations.empty()) {
			auto reservation = reservations[batch].GetValue<int64_t>();
			if (reservation <= 0) {
				throw BinderException("reservations must be positive");
			}
			policy.SetReservation(fixed_bytes, per_worker_bytes, NumericCast<idx_t>(reservation));
		}
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
                        .AddPositionalOnly("batch_bytes", LogicalType::LIST(LogicalType::BIGINT))
                        .WithTypedKwargs("options",
                                         [&](TypedKwargs &options) {
	                                         options.Add("worker_bytes", LogicalType::BIGINT)
	                                             .Add("fixed_bytes", LogicalType::BIGINT)
	                                             .Add("per_worker_bytes", LogicalType::BIGINT)
	                                             .Add("reservations", LogicalType::LIST(LogicalType::BIGINT));
                                         }),
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
