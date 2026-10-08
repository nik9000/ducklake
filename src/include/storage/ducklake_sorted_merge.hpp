//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_sorted_merge.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parallel/thread_context.hpp"
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
};

//! Plans a k-way merge of files that are each sorted on the table's sort key, reading only the key columns
class DuckLakeSortedMergePlanner {
public:
	DuckLakeSortedMergePlanner(ClientContext &context, DuckLakeTableEntry &table,
	                           const vector<unique_ptr<DuckLakeFileScanBind>> &inputs);
	~DuckLakeSortedMergePlanner();

	//! Whether the table has a sort key that can be merged on
	static bool HasMergeableSort(DuckLakeTableEntry &table);
	//! Plan the next (up to) max_rows rows in merged order, returns false once every row has been planned
	bool NextBatch(idx_t max_rows, vector<DuckLakeMergeSlice> &slices);
	//! The file row number of the next row of an input to be planned, invalid once every row of it is planned
	optional_idx NextRow(idx_t input_idx) const;

private:
	struct MergeInput;

	void Initialize();
	bool LoadChunk(MergeInput &input);
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
	//! Columns scanned by the planner: the columns the sort key reads, then file_row_number
	vector<column_t> scan_column_ids;
	vector<unique_ptr<Expression>> key_expressions;
	vector<OrderModifiers> modifiers;
	unique_ptr<ExpressionExecutor> executor;
	vector<unique_ptr<MergeInput>> merge_inputs;
	//! Heap of the inputs that still have rows, ordered by their current row
	vector<idx_t> heap;
	bool initialized = false;
};

} // namespace duckdb
