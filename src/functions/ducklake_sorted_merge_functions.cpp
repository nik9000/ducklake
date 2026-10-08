#include "functions/ducklake_table_functions.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/partition_info.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
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

static vector<unique_ptr<DuckLakeFileScanBind>> BindMergeInputs(ClientContext &context, DuckLakeTableEntry &table,
                                                                const vector<DuckLakeFileListEntry> &files) {
	auto &transaction = DuckLakeTransaction::Get(context, table.ParentCatalog());
	auto file_sort_ids =
	    transaction.GetMetadataManager().ReadDataFileSortIds(table.GetTableId(), transaction.GetSnapshot());
	auto current_sort_id = table.GetSortData()->WrittenSortId();
	vector<unique_ptr<DuckLakeFileScanBind>> result;
	for (auto &file : files) {
		auto input = make_uniq<DuckLakeFileScanBind>(context, table, file);
		auto entry = file_sort_ids.find(file.file_id);
		input->known_sorted =
		    current_sort_id.IsValid() && entry != file_sort_ids.end() && entry->second == current_sort_id.GetIndex();
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
	DuckLakeSortedMergePlanner planner(context, table, inputs);
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
struct SortedMergeBindData : public TableFunctionData {
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
	map<idx_t, unique_ptr<SortedMergeSegmentReader>> segments;
	//! The first row of each admitted batch that has not yet taken its units
	multiset<idx_t> pending_batch_starts;
	//! Rows before this are planned, so no batch that is not yet admitted needs them
	idx_t planned_row = 0;
};

struct SortedMergeGlobalState : public GlobalTableFunctionState {
	mutex lock;
	vector<unique_ptr<DuckLakeFileScanBind>> inputs;
	unique_ptr<DuckLakeSortedMergePlanner> planner;
	vector<unique_ptr<SortedMergeInputReader>> readers;
	idx_t next_batch_index = 0;
	idx_t max_threads = 1;
	//! Decoded bytes a unit grows to before the next one starts
	idx_t unit_size_bytes = 0;
	//! File rows per segment, segments are aligned to row groups of this size
	idx_t segment_size = 0;

	idx_t MaxThreads() const override {
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

struct SortedMergeLocalState : public LocalTableFunctionState {
	idx_t batch_index = 0;
	bool has_batch = false;
	vector<DuckLakeMergeSlice> slices;
	idx_t slice_idx = 0;
	idx_t slice_offset = 0;
	unordered_map<idx_t, SortedMergeInputCursor> cursors;
	vector<LogicalType> payload_types;
	vector<column_t> payload_columns;
	DataChunk payload;
};

static unique_ptr<FunctionData> DuckLakeSortedMergeBind(ClientContext &context, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<Identifier> &names) {
	vector<DuckLakeFileListEntry> files;
	idx_t row_group_size;
	auto &table = BindSortedMergeTable(context, input, files, row_group_size);
	for (auto &column : table.GetColumns().Physical()) {
		names.push_back(column.GetName());
		return_types.push_back(column.Type());
	}
	auto result = make_uniq<SortedMergeBindData>(table);
	result->files = std::move(files);
	result->row_group_size = row_group_size;
	Value decode_unit_size;
	if (DuckLakeTableFunctionUtil::TryGetNonNullOption(input, "decode_unit_size", "integer", decode_unit_size)) {
		result->decode_unit_size = NumericCast<idx_t>(MaxValue<int64_t>(decode_unit_size.GetValue<int64_t>(), 1));
	}
	return std::move(result);
}

//! Units take at most this fraction of the memory limit across all inputs
static constexpr idx_t UNIT_MEMORY_FRACTION = 16;
static constexpr idx_t MIN_UNIT_SIZE_BYTES = 1ULL << 20;
static constexpr idx_t MAX_UNIT_SIZE_BYTES = 32ULL << 20;

static unique_ptr<GlobalTableFunctionState> DuckLakeSortedMergeInitGlobal(ClientContext &context,
                                                                          TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<SortedMergeBindData>();
	auto result = make_uniq<SortedMergeGlobalState>();
	result->inputs = BindMergeInputs(context, bind_data.table, bind_data.files);
	result->planner = make_uniq<DuckLakeSortedMergePlanner>(context, bind_data.table, result->inputs);
	for (idx_t input_idx = 0; input_idx < result->inputs.size(); input_idx++) {
		result->readers.push_back(make_uniq<SortedMergeInputReader>());
	}
	// every input keeps about one unit decoded, so size units to keep that a small part of the memory limit
	auto max_memory = BufferManager::GetBufferManager(context).GetMaxMemory();
	auto unit_size = max_memory / (UNIT_MEMORY_FRACTION * MaxValue<idx_t>(result->inputs.size(), 1));
	result->unit_size_bytes = MaxValue(MIN_UNIT_SIZE_BYTES, MinValue(MAX_UNIT_SIZE_BYTES, unit_size));
	if (bind_data.decode_unit_size.IsValid()) {
		result->unit_size_bytes = bind_data.decode_unit_size.GetIndex();
	}
	result->segment_size = bind_data.row_group_size;
	result->max_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	return std::move(result);
}

static unique_ptr<LocalTableFunctionState> DuckLakeSortedMergeInitLocal(ExecutionContext &context,
                                                                        TableFunctionInitInput &input,
                                                                        GlobalTableFunctionState *global_state) {
	auto &bind_data = input.bind_data->Cast<SortedMergeBindData>();
	auto result = make_uniq<SortedMergeLocalState>();
	for (auto &column : bind_data.table.GetColumns().Physical()) {
		result->payload_columns.push_back(result->payload_types.size());
		result->payload_types.push_back(column.Type());
	}
	result->payload.InitializeEmpty(result->payload_types);
	return std::move(result);
}

static void DecodeNextUnit(ClientContext &context, const SortedMergeBindData &bind_data, SortedMergeGlobalState &gstate,
                           SortedMergeLocalState &lstate, idx_t input_idx, SortedMergeSegmentReader &reader) {
	auto row_number_idx = lstate.payload_columns.size();
	if (!reader.scanner) {
		auto column_ids = lstate.payload_columns;
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
				entry =
				    make_uniq<SortedMergeSegmentReader>(segment_idx * segment_size, (segment_idx + 1) * segment_size);
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

static bool LoadSortedMergeBatch(ClientContext &context, const SortedMergeBindData &bind_data,
                                 SortedMergeGlobalState &gstate, SortedMergeLocalState &lstate) {
	{
		lock_guard<mutex> guard(gstate.lock);
		if (!gstate.planner->NextBatch(bind_data.row_group_size, lstate.slices)) {
			return false;
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
		for (auto &entry : lstate.cursors) {
			auto &reader = *gstate.readers[entry.first];
			lock_guard<mutex> reader_guard(reader.lock);
			reader.pending_batch_starts.insert(entry.second.row_start);
			auto next_row = gstate.planner->NextRow(entry.first);
			reader.planned_row = next_row.IsValid() ? next_row.GetIndex() : NumericLimits<idx_t>::Maximum();
		}
	}
	for (auto &entry : lstate.cursors) {
		TakeUnits(context, bind_data, gstate, lstate, entry.first, entry.second);
	}
	lstate.slice_idx = 0;
	lstate.slice_offset = 0;
	lstate.has_batch = true;
	return true;
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

static void EmitSortedMergeRows(SortedMergeLocalState &lstate, DataChunk &output) {
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	while (output.size() < STANDARD_VECTOR_SIZE && lstate.slice_idx < lstate.slices.size()) {
		auto &slice = lstate.slices[lstate.slice_idx];
		auto &cursor = lstate.cursors[slice.input_idx];
		auto available = PositionCursor(cursor, slice);
		if (lstate.slice_offset == 0 &&
		    cursor.units[cursor.unit_idx]->row_numbers[cursor.unit_row] != slice.row_start) {
			throw InternalException("Sorted merge expected file row %llu of input %llu but found %llu", slice.row_start,
			                        slice.input_idx, cursor.units[cursor.unit_idx]->row_numbers[cursor.unit_row]);
		}
		auto count = MinValue(slice.row_count - lstate.slice_offset, available);
		count = MinValue(count, STANDARD_VECTOR_SIZE - output.size());
		auto chunk_offset = cursor.unit_row - cursor.scan_state.current_row_index;
		for (idx_t i = 0; i < count; i++) {
			sel.set_index(i, chunk_offset + i);
		}
		output.Append(cursor.chunk, sel, count);

		cursor.unit_row += count;
		lstate.slice_offset += count;
		if (lstate.slice_offset == slice.row_count) {
			lstate.slice_idx++;
			lstate.slice_offset = 0;
		}
	}
}

static void DuckLakeSortedMergeExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<SortedMergeBindData>();
	auto &gstate = data.global_state->Cast<SortedMergeGlobalState>();
	auto &lstate = data.local_state->Cast<SortedMergeLocalState>();
	while (output.size() == 0) {
		if (!lstate.has_batch && !LoadSortedMergeBatch(context, bind_data, gstate, lstate)) {
			return;
		}
		EmitSortedMergeRows(lstate, output);
		if (lstate.slice_idx == lstate.slices.size()) {
			// release the decode units of the batch
			lstate.cursors.clear();
			lstate.has_batch = false;
		}
	}
}

static OperatorPartitionData DuckLakeSortedMergePartitionData(ClientContext &context,
                                                              TableFunctionGetPartitionInput &input) {
	return OperatorPartitionData(input.local_state->Cast<SortedMergeLocalState>().batch_index);
}

DuckLakeSortedMergeFunction::DuckLakeSortedMergeFunction()
    : TableFunction("ducklake_sorted_merge", FunctionSignature().AddPositionalOnly("catalog", LogicalType::VARCHAR),
                    DuckLakeSortedMergeExecute, DuckLakeSortedMergeBind, DuckLakeSortedMergeInitGlobal,
                    DuckLakeSortedMergeInitLocal) {
	AddSortedMergeArguments(GetSignature());
	get_partition_data = DuckLakeSortedMergePartitionData;
}

} // namespace duckdb
