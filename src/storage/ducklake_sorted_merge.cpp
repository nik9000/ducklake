#include "storage/ducklake_sorted_merge.hpp"

#include "duckdb/common/bswap.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
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
// Merge Planner
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

struct DuckLakeSortedMergePlanner::MergeInput {
	idx_t input_idx;
	unique_ptr<DuckLakeSingleFileScanner> scanner;
	DataChunk scan_chunk;
	DataChunk key_chunk;
	unique_ptr<Vector> sort_keys;
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
};

DuckLakeSortedMergePlanner::DuckLakeSortedMergePlanner(ClientContext &context, DuckLakeTableEntry &table,
                                                       const vector<unique_ptr<DuckLakeFileScanBind>> &inputs)
    : context(context), table(table), inputs(inputs) {
	auto sort_data = table.GetSortData();
	if (!sort_data) {
		throw InvalidInputException("Table \"%s\" has no sort key to merge on", table.name);
	}
	auto binder = Binder::CreateBinder(context);
	TableIndex table_index(0);
	auto orders = DuckLakeCompactor::BindSortOrders(*binder, table.GetColumns(), table.name, table_index,
	                                                DuckLakeCompactor::ParseSortOrders(*sort_data));
	if (orders.empty()) {
		throw InvalidInputException("Table \"%s\" has no sort key in the DuckDB dialect to merge on", table.name);
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
	for (auto &order : orders) {
		resolver.VisitExpression(&order.expression);
		modifiers.emplace_back(order.type, order.null_order);
		key_expressions.push_back(std::move(order.expression));
	}
	executor = make_uniq<ExpressionExecutor>(context, key_expressions);
}

DuckLakeSortedMergePlanner::~DuckLakeSortedMergePlanner() {
}

bool DuckLakeSortedMergePlanner::HasMergeableSort(DuckLakeTableEntry &table) {
	auto sort_data = table.GetSortData();
	return sort_data && !DuckLakeCompactor::ParseSortOrders(*sort_data).empty();
}

void DuckLakeSortedMergePlanner::Initialize() {
	vector<LogicalType> key_types;
	for (auto &expr : key_expressions) {
		key_types.push_back(expr->GetReturnType());
	}
	for (idx_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
		auto input = make_uniq<MergeInput>();
		input->input_idx = input_idx;
		input->scanner = make_uniq<DuckLakeSingleFileScanner>(context, table, *inputs[input_idx], scan_column_ids);
		input->scan_chunk.Initialize(context, input->scanner->GetTypes());
		input->key_chunk.Initialize(context, key_types);
		merge_inputs.push_back(std::move(input));
	}
	for (auto &input : merge_inputs) {
		if (LoadChunk(*input)) {
			PushInput(input->input_idx);
		}
	}
	initialized = true;
}

bool DuckLakeSortedMergePlanner::LoadChunk(MergeInput &input) {
	if (!input.keys.empty()) {
		input.previous_key_data = input.keys.back().key.GetString();
		input.previous_key = MergeKey(
		    string_t(input.previous_key_data.c_str(), UnsafeNumericCast<uint32_t>(input.previous_key_data.size())));
		input.has_previous_key = true;
	}
	input.keys.clear();
	input.row_numbers.clear();
	input.position = 0;
	if (!input.scanner->Scan(input.scan_chunk)) {
		input.exhausted = true;
		return false;
	}
	auto count = input.scan_chunk.size();
	input.key_chunk.Reset();
	executor->Execute(input.scan_chunk, input.key_chunk);
	input.sort_keys = make_uniq<Vector>(LogicalType::BLOB, count);
	CreateSortKeyHelpers::CreateSortKey(input.key_chunk, modifiers, *input.sort_keys);
	for (auto entry : input.sort_keys->Values<string_t>()) {
		input.keys.emplace_back(entry.GetValue());
	}
	auto &row_number_vector = input.scan_chunk.data[input.scan_chunk.ColumnCount() - 1];
	for (auto entry : row_number_vector.Values<int64_t>()) {
		input.row_numbers.push_back(NumericCast<idx_t>(entry.GetValue()));
	}

	if (inputs[input.input_idx]->known_sorted) {
		return true;
	}
	bool out_of_order = input.has_previous_key && input.keys[0] < input.previous_key;
	for (idx_t i = 1; i < count && !out_of_order; i++) {
		out_of_order = input.keys[i] < input.keys[i - 1];
	}
	if (out_of_order) {
		throw InvalidInputException("Cannot merge file \"%s\": it is not sorted on the sort key of table \"%s\"",
		                            inputs[input.input_idx]->file.file.path, table.name);
	}
	return true;
}

idx_t DuckLakeSortedMergePlanner::RunEnd(const MergeInput &input, idx_t start, idx_t end,
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

bool DuckLakeSortedMergePlanner::InputBefore(const MergeInput &lhs, const MergeInput &rhs) const {
	auto &lhs_key = lhs.CurrentKey();
	auto &rhs_key = rhs.CurrentKey();
	if (lhs_key < rhs_key) {
		return true;
	}
	return lhs.input_idx < rhs.input_idx && lhs_key == rhs_key;
}

void DuckLakeSortedMergePlanner::PushInput(idx_t input_idx) {
	heap.push_back(input_idx);
	std::push_heap(heap.begin(), heap.end(),
	               [&](idx_t lhs, idx_t rhs) { return InputBefore(*merge_inputs[rhs], *merge_inputs[lhs]); });
}

void DuckLakeSortedMergePlanner::SiftDownTop() {
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

optional_idx DuckLakeSortedMergePlanner::NextRow(idx_t input_idx) const {
	auto &input = *merge_inputs[input_idx];
	if (input.exhausted) {
		return optional_idx();
	}
	return input.row_numbers[input.position];
}

bool DuckLakeSortedMergePlanner::NextBatch(idx_t max_rows, vector<DuckLakeMergeSlice> &slices) {
	if (!initialized) {
		Initialize();
	}
	slices.clear();
	idx_t planned = 0;
	while (planned < max_rows && !heap.empty()) {
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
			auto end = MinValue(input.keys.size(), input.position + max_rows - planned);
			auto run_end = RunEnd(input, MinValue(input.position + known_rows, end), end, next_input);
			known_rows = 0;
			if (run_end > input.position) {
				slice.row_end = input.row_numbers[run_end - 1] + 1;
				slice.row_count += run_end - input.position;
				planned += run_end - input.position;
				input.position = run_end;
			}
			if (input.position < input.keys.size() || !LoadChunk(input) || planned == max_rows) {
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
