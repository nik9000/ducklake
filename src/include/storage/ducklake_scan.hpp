//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/catalog/catalog_entry/copy_function_catalog_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/function/table_function.hpp"
#include "common/ducklake_snapshot.hpp"
#include "common/index.hpp"
#include "storage/ducklake_metadata_info.hpp"

namespace duckdb {
class DuckLakeMultiFileList;
class DuckLakeTableEntry;
class DuckLakeTransaction;
class Serializer;
class Deserializer;

class DuckLakeFunctions {
public:
	//! Table Functions
	static TableFunction GetDuckLakeScanFunction(DatabaseInstance &instance);

	static unique_ptr<FunctionData> BindDuckLakeScan(ClientContext &context, TableFunction &function);

	static CopyFunctionCatalogEntry &GetCopyFunction(ClientContext &context, const Identifier &name);
};

//! Serialize/Deserialize callbacks for DuckLakeScan (used by table macro Copy)
void DuckLakeScanSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
                           const TableFunction &function);
unique_ptr<FunctionData> DuckLakeScanDeserialize(Deserializer &deserializer, TableFunction &function);

enum class DuckLakeScanType { SCAN_TABLE, SCAN_INSERTIONS, SCAN_DELETIONS, SCAN_FOR_FLUSH };

//! The transaction-local state a scan is allowed to observe. A scan pinned to a historical snapshot through an AT
//! clause reads a fully-committed view, so it holds no transaction and every query below reports "nothing".
class DuckLakeScanLocalChanges {
public:
	DuckLakeScanLocalChanges() = default;
	explicit DuckLakeScanLocalChanges(shared_ptr<DuckLakeTransaction> transaction_p);

	vector<DuckLakeDataFile> GetFiles(TableIndex table_id) const;
	shared_ptr<DuckLakeInlinedData> GetInlinedData(TableIndex table_id) const;
	bool HasDroppedFiles() const;
	bool FileIsDropped(const string &path) const;
	bool HasDeletes(TableIndex table_id) const;
	bool HasDeleteForFile(TableIndex table_id, const string &path) const;
	void GetDeleteForFile(TableIndex table_id, const string &path, DuckLakeFileData &result) const;
	bool HasInlinedFileDeletes(TableIndex table_id) const;
	void GetInlinedFileDeletesForFile(TableIndex table_id, idx_t file_id, set<idx_t> &result) const;
	optional_ptr<DuckLakeInlinedDataDeletes> GetInlinedDeletes(TableIndex table_id, const string &table_name) const;

private:
	shared_ptr<DuckLakeTransaction> transaction;
};

struct DuckLakeFunctionInfo : public TableFunctionInfo {
	DuckLakeFunctionInfo(DuckLakeTableEntry &table, DuckLakeTransaction &transaction, DuckLakeSnapshot snapshot);

	static shared_ptr<DuckLakeFunctionInfo> Create(DuckLakeTableEntry &table, DuckLakeTransaction &transaction,
	                                               DuckLakeSnapshot snapshot);

	DuckLakeTableEntry &table;
	weak_ptr<DuckLakeTransaction> transaction;
	string table_name;
	vector<string> column_names;
	vector<LogicalType> column_types;
	DuckLakeSnapshot snapshot;
	TableIndex table_id;
	DuckLakeScanType scan_type = DuckLakeScanType::SCAN_TABLE;
	//! Start snapshot - only set for DuckLakeScanType::SCAN_INSERTIONS and DuckLakeScanType::SCAN_DELETIONS
	unique_ptr<DuckLakeSnapshot> start_snapshot;
	//! Should this scan should include the current transaction's uncommitted local changes?
	//! False when the scan is pinned to a committed, historical snapshot with AT.
	bool include_transaction_local_changes = true;

	shared_ptr<DuckLakeTransaction> GetTransaction();
	//! The transaction-local state this scan may observe - empty for a scan pinned with AT
	DuckLakeScanLocalChanges GetVisibleLocalChanges();
	//! As above, reusing a transaction the caller already holds
	DuckLakeScanLocalChanges GetVisibleLocalChanges(const shared_ptr<DuckLakeTransaction> &transaction);
	bool CanUseGlobalStats();
};

} // namespace duckdb
