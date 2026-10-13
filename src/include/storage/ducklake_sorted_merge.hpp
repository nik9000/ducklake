//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_sorted_merge.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/atomic.hpp"
#include <condition_variable>
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
//! waiting to be written in order. The workers' buffers for ordering and emitting batches take another quarter.
class DuckLakeSortedMergeWorkerPolicy {
public:
	DuckLakeSortedMergeWorkerPolicy(idx_t max_threads, idx_t worker_bytes);

	//! Record the decoded bytes of a full batch
	void AddBatch(idx_t batch_bytes, idx_t max_memory);
	//! Cap the workers to those that fit the merge's memory reservation: fixed_bytes for the merge and worker_bytes
	//! for each worker. The reservation replaces the share of the memory limit for the workers' buffers.
	void SetReservation(idx_t fixed_bytes, idx_t per_worker_bytes, idx_t reservation);
	//! The workers allowed under the current memory limit
	idx_t WorkerLimit(idx_t max_memory) const;
	idx_t AverageBatchBytes() const {
		return average_batch_bytes;
	}

private:
	idx_t TargetWorkers(idx_t max_memory) const;

private:
	idx_t max_threads;
	//! The buffers each worker uses to order and emit a batch
	idx_t worker_bytes;
	//! The merge's memory reservation, when one is made
	idx_t reservation = 0;
	idx_t reservation_fixed_bytes = 0;
	idx_t reservation_worker_bytes = 0;
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

//! The next row_count rows of one merge input in merged order, each input's rows following on from its previous run
struct DuckLakeMergeRun {
	uint32_t input_idx;
	uint32_t row_count;
};

//! A growable array of trivially copyable values in memory from an allocator, which keeps its memory when cleared
template <class T>
class DuckLakeAllocatedArray {
public:
	explicit DuckLakeAllocatedArray(Allocator &allocator) : allocator(&allocator) {
	}

	idx_t size() const {
		return count;
	}
	bool empty() const {
		return count == 0;
	}
	T *data() {
		return reinterpret_cast<T *>(buffer.get());
	}
	const T *data() const {
		return reinterpret_cast<const T *>(buffer.get());
	}
	T &operator[](idx_t idx) {
		return data()[idx];
	}
	const T &operator[](idx_t idx) const {
		return data()[idx];
	}
	T &back() {
		return data()[count - 1];
	}
	T *begin() {
		return data();
	}
	T *end() {
		return data() + count;
	}
	const T *begin() const {
		return data();
	}
	const T *end() const {
		return data() + count;
	}
	void push_back(const T &value) {
		if (count == capacity) {
			Grow(MaxValue<idx_t>(capacity * 2, MIN_CAPACITY));
		}
		data()[count++] = value;
	}
	void reserve(idx_t new_capacity) {
		if (new_capacity > capacity) {
			Grow(new_capacity);
		}
	}
	//! Sets the size without initializing new values
	void resize(idx_t new_count) {
		if (new_count > capacity) {
			Grow(new_count);
		}
		count = new_count;
	}
	void clear() {
		count = 0;
	}
	//! Clears the array and gives its memory back to the allocator
	void Release() {
		buffer.Reset();
		count = 0;
		capacity = 0;
	}
	void Swap(DuckLakeAllocatedArray &other) {
		std::swap(allocator, other.allocator);
		std::swap(buffer, other.buffer);
		std::swap(count, other.count);
		std::swap(capacity, other.capacity);
	}

private:
	static constexpr idx_t MIN_CAPACITY = 1024;

	void Grow(idx_t new_capacity) {
		auto new_buffer = allocator->Allocate(new_capacity * sizeof(T));
		if (count > 0) {
			memcpy(new_buffer.get(), buffer.get(), count * sizeof(T));
		}
		buffer = std::move(new_buffer);
		capacity = new_capacity;
	}

	Allocator *allocator;
	AllocatedData buffer;
	idx_t count = 0;
	idx_t capacity = 0;
};

//! The bytes of the sort keys of a chunk that do not fit in a string_t
struct DuckLakeKeyData {
	AllocatedData data;
};

//! The sort keys of sequenced rows in merged order, sharing the sequencer's key data
struct DuckLakeMergeKeys {
	explicit DuckLakeMergeKeys(Allocator &allocator) : keys(allocator) {
	}

	DuckLakeAllocatedArray<string_t> keys;
	//! Keep the key data alive
	vector<shared_ptr<DuckLakeKeyData>> owners;

	void Clear() {
		keys.clear();
		owners.clear();
	}
};

//! The buffers a worker orders the rows of a batch in
struct DuckLakeOrderBuffers;

struct DuckLakeSortedMergeKeyChunk;

//! The rows of a batch chosen by the sequencer, put in merged order by OrderBatch
struct DuckLakeSequencedBatch {
	//! Consecutive rows of one input: row_count rows from row offset of the first chunk on
	struct Range {
		idx_t input_idx = 0;
		vector<shared_ptr<DuckLakeSortedMergeKeyChunk>> chunks;
		idx_t offset = 0;
		idx_t row_count = 0;
	};
	//! The ranges of each section the batch reads, in the order of the sections, merged within a section
	vector<vector<Range>> parts;
	idx_t row_count = 0;

	void Clear() {
		parts.clear();
		row_count = 0;
	}
};

//! Sequences a k-way merge of files that are each sorted on the table's sort key: which input each run of output rows
//! comes from, reading only the key columns
class DuckLakeSortedMergeSequencer {
public:
	static constexpr idx_t MIN_CHUNKS_AHEAD = 4;
	static constexpr idx_t MAX_CHUNKS_AHEAD = 64;

	//! Inputs are read in segments of segment_size rows
	DuckLakeSortedMergeSequencer(ClientContext &context, DuckLakeTableEntry &table,
	                             const vector<unique_ptr<DuckLakeFileScanBind>> &inputs, idx_t segment_size);
	~DuckLakeSortedMergeSequencer();

	//! Whether the table has a sort key that can be merged on
	static bool HasMergeableSort(DuckLakeTableEntry &table);
	//! Read the first keys of the inputs and group them into sections, before the first batch
	void Initialize();
	//! Choose the next (up to) max_rows rows of the merge, returns false once every row has been sequenced. Within a
	//! section of several inputs the batch is the rows before a cut in each input, found by multisequence selection.
	bool NextBatch(idx_t max_rows, DuckLakeSequencedBatch &batch);
	//! Put the rows of a batch in merged order, without the lock that guards NextBatch. When keys is given, it receives
	//! the sort keys of the rows in merged order.
	static void OrderBatch(const DuckLakeSequencedBatch &batch, DuckLakeOrderBuffers &buffers,
	                       DuckLakeAllocatedArray<DuckLakeMergeRun> &runs,
	                       optional_ptr<DuckLakeMergeKeys> keys = nullptr);
	static shared_ptr<DuckLakeOrderBuffers> CreateOrderBuffers(Allocator &allocator);
	//! Gives the memory of the buffers back to their allocator
	static void ReleaseOrderBuffers(DuckLakeOrderBuffers &buffers);
	//! The most memory a worker's buffers take to order and emit a batch of batch_rows rows
	static idx_t WorkerBytes(idx_t batch_rows, bool with_keys);
	//! The file rows of the runs of a batch in merged order
	static void RunSlices(const DuckLakeSequencedBatch &batch, const DuckLakeAllocatedArray<DuckLakeMergeRun> &runs,
	                      vector<DuckLakeMergeSlice> &slices);
	//! The rows of every input a batch reads, one slice per input
	static void BatchInputRows(const DuckLakeSequencedBatch &batch, vector<DuckLakeMergeSlice> &input_rows);
	//! Read about a batch's share of keys of each input ahead, for batches of batch_rows rows
	void SetBatchRows(idx_t batch_rows);
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
	//! Workers with nothing to load wait for work: the merge taking loaded keys of an input, or the lock that
	//! guards NextBatch being released. A worker reads the generation before Prefetch, so no notification is lost.
	idx_t WaitGeneration() const {
		return wait_generation;
	}
	void WaitForWork(idx_t seen_generation);
	//! Whether the section being merged merges keys of several inputs, so workers can load keys ahead for it
	bool MergingKeys() const;
	//! Wake a waiting worker, or all of them
	void NotifyWaiters(bool all = false);
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
	//! The most segments of sort keys loaded at once, each with its own scanner
	idx_t MaxSegmentsLoaded() const;
	//! The chunks of keys a segment loads ahead
	idx_t ChunksAhead() const {
		return chunks_ahead;
	}
	//! The memory of the sort keys and row number of a row, apart from the bytes of the key
	static idx_t KeyRowBytes();

private:
	struct KeySegment;
	struct MergeInput;
	struct WindowSequences;

	//! Read the key of the input's last row
	void ReadLastKey(MergeInput &input);
	//! Bound the input's keys by the catalog's statistics of the sort key columns, returns false when they do not
	bool CatalogBounds(const map<idx_t, DuckLakeFileKeyStats> &file_stats, const vector<FieldIndex> &fields,
	                   MergeInput &input);
	//! Read only the row numbers of an input merged alone from now on
	//! Sets each input's key prefix from the bounds of its section
	void SetKeyPrefixes();
	void MakeKeyless(MergeInput &input);
	//! Start merging the inputs of the next section, returns false when every section is merged
	bool StartNextSection();
	//! Take the input's next chunk of keys into its window, returns false when every chunk is taken
	bool TakeChunk(MergeInput &input);
	//! Throw when the chunk's keys are not sorted, unless the catalog records the input as sorted in a release build
	void VerifySorted(const MergeInput &input, const DuckLakeSortedMergeKeyChunk &chunk);
	//! Move the next count rows of the input's window into the range
	void TakeRows(MergeInput &input, idx_t count, DuckLakeSequencedBatch::Range &range);
	//! The loaded rows of the section that come before or at the last loaded row of the frontier input
	idx_t RowsUpTo(const MergeInput &frontier) const;
	//! Choose the next (up to) max_rows rows of a section of several inputs
	void SequenceSection(idx_t max_rows, vector<DuckLakeSequencedBatch::Range> &part);
	//! Load the next chunk of keys of a segment, the caller holds the segment's lock
	void LoadKeys(MergeInput &input, KeySegment &segment);
	//! Load a chunk of keys into a segment that is not yet loaded far enough ahead
	bool TopUpSegment(MergeInput &input);
	//! Start loading the next segment of the input with the fewest segments loaded ahead
	bool StartSegment();

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
	vector<unique_ptr<MergeInput>> merge_inputs;
	idx_t segment_size;
	atomic<idx_t> max_segments_ahead {0};
	//! Segments created and not yet merged, across inputs
	atomic<idx_t> segments_ahead {0};
	//! Where the next Prefetch starts looking, so inputs are visited in turn
	atomic<idx_t> prefetch_cursor {0};
	mutex waiters_lock;
	std::condition_variable waiters;
	atomic<idx_t> wait_generation {0};
	//! The inputs of the section being merged that have rows left, by input
	vector<idx_t> section_inputs;
	//! Chunks of keys each segment loads ahead of the merge
	atomic<idx_t> chunks_ahead {MIN_CHUNKS_AHEAD};
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
