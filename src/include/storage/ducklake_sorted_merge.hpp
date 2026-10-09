//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_sorted_merge.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/table_filter_set.hpp"
#include "storage/ducklake_metadata_info.hpp"

namespace duckdb {
class DuckLakeTableEntry;

//! The DuckLake scan function bound to a single data file
struct DuckLakeFileScanBind {
	DuckLakeFileScanBind(ClientContext &context, DuckLakeTableEntry &table, const DuckLakeFileListEntry &file);

	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	DuckLakeFileListEntry file;
	//! Whether the catalog records the file as sorted on the table's current sort, so it needs no verification
	bool known_sorted = false;
	//! The rows in the file, deleted rows included, when the catalog records them
	optional_idx row_count;
};

//! Scans one DuckLake data file in file order, applying field-id mapping and deletes like a regular scan
class DuckLakeSingleFileScanner {
public:
	DuckLakeSingleFileScanner(ClientContext &context, DuckLakeTableEntry &table, const DuckLakeFileScanBind &scan_bind,
	                          vector<column_t> column_ids, unique_ptr<TableFilterSet> filters = nullptr);

	const vector<LogicalType> &GetTypes() const {
		return types;
	}
	//! Scan the next chunk, returns false when the file is exhausted
	bool Scan(DataChunk &chunk);

private:
	ClientContext &context;
	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> types;
	unique_ptr<TableFilterSet> filters;
	unique_ptr<ThreadContext> thread_context;
	unique_ptr<ExecutionContext> execution_context;
	unique_ptr<GlobalTableFunctionState> global_state;
	unique_ptr<LocalTableFunctionState> local_state;
};

//! Chooses how many workers of a sorted merge may emit batches at once, from the decoded size of the batches. The batch
//! copy takes a quarter of the memory limit and each worker keeps about two batches in it: one it builds and one
//! waiting to be written in order.
class DuckLakeSortedMergeWorkerPolicy {
public:
	explicit DuckLakeSortedMergeWorkerPolicy(idx_t max_threads);

	//! Record the decoded bytes of a full batch
	void AddBatch(idx_t batch_bytes, idx_t max_memory);
	//! The workers allowed under the current memory limit
	idx_t WorkerLimit(idx_t max_memory) const;
	idx_t AverageBatchBytes() const {
		return average_batch_bytes;
	}

private:
	idx_t TargetWorkers(idx_t max_memory) const;

private:
	idx_t max_threads;
	//! Follows wider batches at once and narrower batches slowly
	idx_t average_batch_bytes = 0;
	//! Drops at once and at most doubles per batch
	idx_t limit = 1;
};

//! A run of rows from one merge input: the live rows among file rows [row_start, row_end)
struct DuckLakeMergeSlice {
	idx_t input_idx;
	idx_t row_start;
	idx_t row_end;
	idx_t row_count;
	//! Whether the sequencer read the sort keys of the rows
	bool has_keys = true;
};

//! The sort keys of sequenced rows in merged order, sharing the sequencer's key data
struct DuckLakeMergeKeys {
	vector<string_t> keys;
	//! Keep the key data alive
	vector<shared_ptr<Vector>> owners;
};

//! Sequences a k-way merge of files that are each sorted on the table's sort key: which input each run of output rows
//! comes from, reading only the key columns
class DuckLakeSortedMergeSequencer {
public:
	//! Inputs are read in segments of segment_size rows
	DuckLakeSortedMergeSequencer(ClientContext &context, DuckLakeTableEntry &table,
	                             const vector<unique_ptr<DuckLakeFileScanBind>> &inputs, idx_t segment_size);
	~DuckLakeSortedMergeSequencer();

	//! Whether the table has a sort key that can be merged on
	static bool HasMergeableSort(DuckLakeTableEntry &table);
	//! Read the first keys of the inputs and group them into sections, before the first batch
	void Initialize();
	//! Sequence the next (up to) max_rows rows in merged order, returns false once every row has been sequenced. When
	//! keys is given, it receives the sort keys of the sequenced rows in merged order.
	bool NextBatch(idx_t max_rows, vector<DuckLakeMergeSlice> &slices, optional_ptr<DuckLakeMergeKeys> keys = nullptr);
	//! The table columns of the sort key in order, when every sort expression is a distinct column, so their values can
	//! be decoded from the sort keys. Empty otherwise.
	const vector<idx_t> &DecodableColumns() const {
		// rows merged without their keys read the key columns from the files
		return keyless_sections > 0 ? no_columns : decodable_columns;
	}
	const vector<OrderModifiers> &Modifiers() const {
		return modifiers;
	}
	//! The file row number of the next row of an input to be sequenced, invalid once every row of it is sequenced
	optional_idx NextRow(idx_t input_idx) const;
	//! Load sort keys ahead of the merge. Safe to call without holding the lock that guards NextBatch. Returns false
	//! when there is nothing to load.
	bool Prefetch();
	//! Whether the input is merged in the section being merged or the next one, so reading ahead for it pays off soon
	bool InputIsNear(idx_t input_idx) const;
	//! The number of sections of inputs whose key ranges overlap, each merged after the one before
	idx_t SectionCount() const {
		return sections.size();
	}
	idx_t KeylessSectionCount() const {
		return keyless_sections;
	}
	//! At most this many segments, across inputs, are loaded at once. Prefetch loads none ahead until it is set.
	void SetMaxSegmentsAhead(idx_t max_segments) {
		max_segments_ahead = max_segments;
	}

private:
	struct KeyChunk;
	struct KeySegment;
	struct MergeInput;

	//! Read the key of the input's last row
	void ReadLastKey(MergeInput &input);
	//! Bound the input's keys by the catalog's statistics of the sort key columns, returns false when they do not
	bool CatalogBounds(const map<idx_t, DuckLakeFileKeyStats> &file_stats, const vector<FieldIndex> &fields,
	                   MergeInput &input);
	//! Read only the row numbers of an input merged alone from now on
	void MakeKeyless(MergeInput &input);
	//! Start merging the inputs of the next section, returns false when every section is merged
	bool StartNextSection();
	bool LoadChunk(MergeInput &input);
	//! Load the next chunk of keys of a segment, the caller holds the segment's lock
	void LoadKeys(MergeInput &input, KeySegment &segment);
	//! Load a chunk of keys into a segment that is not yet loaded far enough ahead
	bool TopUpSegment(MergeInput &input);
	//! Start loading the next segment of the input with the fewest segments loaded ahead
	bool StartSegment();
	//! Whether the current row of lhs comes before the current row of rhs in the merged order
	bool InputBefore(const MergeInput &lhs, const MergeInput &rhs) const;
	//! The end of the run of rows in [input.position, end) that come before the current row of next, given that
	//! the rows before start do
	idx_t RunEnd(const MergeInput &input, idx_t start, idx_t end, optional_ptr<MergeInput> next) const;
	void PushInput(idx_t input_idx);
	//! Restore heap order after the top input's current row changed
	void SiftDownTop();

private:
	ClientContext &context;
	DuckLakeTableEntry &table;
	const vector<unique_ptr<DuckLakeFileScanBind>> &inputs;
	//! Columns scanned by the sequencer: the columns the sort key reads, then file_row_number
	vector<column_t> scan_column_ids;
	vector<unique_ptr<Expression>> key_expressions;
	vector<LogicalType> key_types;
	vector<OrderModifiers> modifiers;
	vector<idx_t> decodable_columns;
	vector<idx_t> no_columns;
	//! The key data each input's keys in the batch being sequenced share, so it is kept alive once
	vector<const Vector *> key_owners;
	vector<unique_ptr<MergeInput>> merge_inputs;
	idx_t segment_size;
	atomic<idx_t> max_segments_ahead {0};
	//! Segments created and not yet merged, across inputs
	atomic<idx_t> segments_ahead {0};
	//! Where the next Prefetch starts looking, so inputs are visited in turn
	atomic<idx_t> prefetch_cursor {0};
	//! Heap of the inputs that still have rows, ordered by their current row
	vector<idx_t> heap;
	atomic<bool> initialized {false};
	//! Inputs grouped into sections of overlapping key ranges, in key order
	vector<vector<idx_t>> sections;
	//! Sections of a single input recorded as sorted, merged without their keys
	idx_t keyless_sections = 0;
	//! Columns scanned for an input merged without its keys: file_row_number
	vector<column_t> row_number_column_ids {MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
	idx_t next_section = 0;
	atomic<idx_t> active_section {0};
};

//! A logical operator emitting the rows of files that are each sorted on the table's sort key, in merged order: the
//! table's physical columns, then each row's snapshot id when write_snapshot_id is set. Its column bindings use
//! table_index.
unique_ptr<LogicalOperator> PlanDuckLakeSortedMerge(DuckLakeTableEntry &table, vector<DuckLakeFileListEntry> files,
                                                    TableIndex table_index, idx_t row_group_size,
                                                    bool write_snapshot_id);

} // namespace duckdb
