#!/bin/bash
# Generates a DuckLake for benchmarking compaction of sorted tables: one table per profile and shape, each written as
# FILES sorted inserts of ROWS rows, so every insert is one data file stamped with the table's sort.
#
# usage: bench_sorted_merge_generate.sh <dir> [profiles] [shapes]
#   profiles: high (about 50:1 compression), md5 (low compression), skinny (sorted on every column)
#   shapes:   disjoint, small_overlap, tenant_time, random
# env: DUCKDB (binary, default build/release/duckdb), ROWS (default 2000000), FILES (default 10)
#
# Tables are named <profile>_<shape> in <dir>/lake.db, with data in <dir>/lake_files.
set -euo pipefail

DIR=$1
PROFILES=${2:-"high md5 skinny"}
SHAPES=${3:-"disjoint small_overlap tenant_time random"}
REPO=$(cd "$(dirname "$0")" && pwd)
DUCKDB=${DUCKDB:-$REPO/build/release/duckdb}
ROWS=${ROWS:-2000000}
FILES=${FILES:-10}

mkdir -p "$DIR"
cd "$DIR"

# the sort key columns of a shape and their values for row r of file i
key_columns() {
	case $1 in
	tenant_time) echo "tenant VARCHAR, ts TIMESTAMP" ;;
	*) echo "k BIGINT" ;;
	esac
}
key_values() {
	local n=$ROWS
	case $1 in
	# every file covers its own key range
	disjoint) echo "i * $n + r AS k" ;;
	# like disjoint, but each file starts 5% before the previous one ends
	small_overlap) echo "i * ($n - $n // 20) + r AS k" ;;
	# every file holds all tenants for a later time window
	tenant_time)
		echo "'tenant-' || lpad((hash(r, i, 't') % 1000)::VARCHAR, 4, '0') AS tenant," \
			"TIMESTAMP '2026-01-01' + to_seconds(i * $n + r) AS ts"
		;;
	# every file covers the whole key range
	random) echo "(hash(r, i, 'k') % 1000000000000)::BIGINT AS k" ;;
	*)
		echo "unknown shape $1" >&2
		exit 1
		;;
	esac
}
sort_key() {
	case $1 in
	tenant_time) echo "tenant, ts" ;;
	*) echo "k" ;;
	esac
}

# the payload columns of a profile and their values, from h = hash(r, i)
payload_columns() {
	case $1 in
	high)
		echo ", user_agent VARCHAR, url VARCHAR, country VARCHAR, status INTEGER, payload VARCHAR, amount DOUBLE," \
			"user_id BIGINT"
		;;
	md5) echo ", a VARCHAR, b VARCHAR, amount DOUBLE" ;;
	skinny) echo ", j INTEGER" ;;
	*)
		echo "unknown profile $1" >&2
		exit 1
		;;
	esac
}
payload_values() {
	case $1 in
	high)
		echo ", 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/' || (h % 50) ||" \
			"'.0.0.0 Safari/537.36 ' || repeat('ext ', 20)," \
			"'https://shop.example.com/catalog/category-' || ((h >> 8) % 1000) || '/item?ref=' || repeat('z', 30)," \
			"'country-' || ((h >> 20) % 30), [200, 301, 404, 500][(((h >> 25) % 4) + 1)::BIGINT]," \
			"'{\"campaign\":\"' || ((h >> 30) % 200) || '\",\"attributes\":\"' || repeat('lorem ipsum ', 30) || '\"}'," \
			"((h >> 40) % 100) * 1.5, ((h >> 12) % 1000000)::BIGINT"
		;;
	md5) echo ", md5(h::VARCHAR), md5((h >> 7)::VARCHAR), ((h >> 40) % 100000) / 100.0" ;;
	skinny) echo ", (h % 1000)::INTEGER" ;;
	esac
}

SQL="ATTACH 'ducklake:lake.db' AS lake (DATA_PATH 'lake_files', DATA_INLINING_ROW_LIMIT 0);"
for profile in $PROFILES; do
	for shape in $SHAPES; do
		table=${profile}_${shape}
		sort=$(sort_key "$shape")
		# the skinny profile sorts on every column
		[ "$profile" = skinny ] && [ "$shape" != tenant_time ] && sort="$sort, j"
		[ "$profile" = skinny ] && [ "$shape" = tenant_time ] && payload_cols="" || payload_cols=$(payload_columns "$profile")
		[ "$profile" = skinny ] && [ "$shape" = tenant_time ] && payload_vals="" || payload_vals=$(payload_values "$profile")
		SQL+="
DROP TABLE IF EXISTS lake.$table;
CREATE TABLE lake.$table ($(key_columns "$shape")$payload_cols);
ALTER TABLE lake.$table SET SORTED BY ($sort);"
		for ((i = 0; i < FILES; i++)); do
			SQL+="
INSERT INTO lake.$table SELECT $(key_values "$shape")$payload_vals
FROM (SELECT $i AS i, r, hash(r, $i) AS h FROM range($ROWS) t(r));"
		done
	done
done
SQL+="
SELECT t.table_name, count(*) AS files, sum(d.record_count) AS rows, round(sum(d.file_size_bytes) / 1e6) AS mb
FROM __ducklake_metadata_lake.ducklake_data_file d
JOIN __ducklake_metadata_lake.ducklake_table t USING (table_id)
WHERE d.end_snapshot IS NULL AND t.end_snapshot IS NULL
GROUP BY ALL ORDER BY ALL;"

"$DUCKDB" -c "$SQL"
