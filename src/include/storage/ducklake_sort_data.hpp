//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_sort_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "common/index.hpp"
#include "duckdb/common/enums/order_type.hpp"

namespace duckdb {

class DuckLakeTableEntry;
struct OrderByNode;

struct DuckLakeSortField {
	idx_t sort_key_index = 0;
	string expression;
	string dialect;
	OrderType sort_direction;
	OrderByNullType null_order;
};

struct DuckLakeSort {
	idx_t sort_id = 0;
	//! The transaction-local id that files written under this sort carry until commit
	optional_idx local_sort_id;
	vector<DuckLakeSortField> fields;

	//! The sort id to record on a file written sorted by this sort, invalid if writes cannot apply every field
	optional_idx WrittenSortId() const;

	//! Build a SQL ORDER BY clause from the parsed sort orders, mapping inlined columns
	static string BuildSortOrderSQL(const vector<OrderByNode> &orders, const DuckLakeTableEntry &current_table,
	                                const DuckLakeTableEntry &inlined_table);
};

} // namespace duckdb
