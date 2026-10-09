#!/bin/bash
# Compacts tables of a lake made by bench_sorted_merge_generate.sh with merge_adjacent_files, merging sorted files and
# sorting them, at several memory limits. Every run starts from the generated catalog and removes the files it wrote.
#
# usage: run.sh <dir> [tables] [limits] [modes]
#   tables: default every table in the lake; limits: default "20GB 8GB 2GB 1GB 500MB"; modes: default "merge sort"
# env: DUCKDB (binary, default build/release/duckdb), REPS (default 2), THREADS (default 24),
#      DATA_PATH (read the lake from elsewhere, e.g. an s3:// copy of <dir>/lake_files/; written files are removed with
#      the aws CLI), TEMP_DIR (spill directory, default <dir>/spill)
#
# Prints one line per run: table limit rep mode seconds peak_rss_mb peak_spill_mb merged files stamped
# where merged says whether the sorted merge ran, and files/stamped count the table's files after the compaction
# and those stamped with its sort. A failed run prints ERR and the error.
set -uo pipefail

DIR=$(cd "$1" && pwd)
REPO=$(cd "$(dirname "$0")" && pwd)
DUCKDB=${DUCKDB:-$REPO/build/release/duckdb}
LIMITS=${3:-"20GB 8GB 2GB 1GB 500MB"}
MODES=${4:-"merge sort"}
REPS=${REPS:-2}
THREADS=${THREADS:-24}
TEMP_DIR=${TEMP_DIR:-$DIR/spill}
DATA_PATH=${DATA_PATH:-}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cd "$DIR"
cp lake.db "$WORK/pristine.db"
TABLES=${2:-$("$DUCKDB" -list -noheader -c "ATTACH 'ducklake:$WORK/pristine.db' AS lake (READ_ONLY);
	SELECT table_name FROM duckdb_tables() WHERE database_name = 'lake' ORDER BY ALL")}

if [ -n "$DATA_PATH" ]; then
	ATTACH="CREATE SECRET (TYPE s3, PROVIDER credential_chain);
		ATTACH 'ducklake:$WORK/work.db' AS lake (DATA_PATH '$DATA_PATH', OVERRIDE_DATA_PATH true);"
	list_files() { aws s3 ls --recursive "$DATA_PATH" | awk '{print $4}' | sort; }
	bucket=$(echo "$DATA_PATH" | cut -d/ -f3)
	remove_file() { aws s3 rm --quiet "s3://$bucket/$1"; }
else
	ATTACH="ATTACH 'ducklake:$WORK/work.db' AS lake;"
	list_files() { find lake_files -type f | sort; }
	remove_file() { rm -f "$1"; }
fi
list_files > "$WORK/pristine_files.txt"

table_files="FROM __ducklake_metadata_lake.ducklake_data_file d
	JOIN __ducklake_metadata_lake.ducklake_table t USING (table_id)
	WHERE t.table_name = getvariable('tbl') AND d.end_snapshot IS NULL AND t.end_snapshot IS NULL"

run() {
	local table=$1 limit=$2 mode=$3
	cp "$WORK/pristine.db" "$WORK/work.db"
	rm -rf "$TEMP_DIR" && mkdir -p "$TEMP_DIR"
	local setup="$ATTACH SET VARIABLE tbl = '$table';"
	if [ "$mode" = sort ]; then
		"$DUCKDB" -c "$setup CALL lake.set_option('sorted_merge_compaction', false, table_name => '$table')" > /dev/null 2>&1
	fi
	/usr/bin/time -o "$WORK/time" -f "%e %M" "$DUCKDB" -c "$setup SET threads = $THREADS;
		SET memory_limit = '$limit'; SET temp_directory = '$TEMP_DIR'; CALL enable_logging(level = 'debug');
		CALL ducklake_merge_adjacent_files('lake', '$table');
		SELECT 'CHECK', (SELECT count(*) $table_files), (SELECT count(d.sort_id) $table_files)" > "$WORK/out" 2>&1 &
	local pid=$! peak=0 spill
	while kill -0 $pid 2> /dev/null; do
		spill=$(du -sm "$TEMP_DIR" 2> /dev/null | cut -f1)
		[ "${spill:-0}" -gt "$peak" ] && peak=$spill
		sleep 0.2
	done
	if grep -q "CHECK" "$WORK/out"; then
		local check merged
		check=$(grep -o 'CHECK[^0-9]*[0-9]*[^0-9]*[0-9]*' "$WORK/out" | grep -o '[0-9]\+' | tr '\n' ' ')
		merged=$(grep -q 'DuckLake sorted merge:' "$WORK/out" && echo yes || echo no)
		echo "$(awk '{printf "%.2f %d", $1, $2 / 1024}' "$WORK/time") $peak $merged $check"
	else
		echo "ERR $(grep -m1 -iE 'error|exception' "$WORK/out" | cut -c1-120) spill=$peak"
	fi
	list_files | comm -13 "$WORK/pristine_files.txt" - | while read -r file; do remove_file "$file"; done
}

for table in $TABLES; do
	for limit in $LIMITS; do
		for ((rep = 1; rep <= REPS; rep++)); do
			for mode in $MODES; do
				echo "$table $limit $rep $mode $(run "$table" "$limit" "$mode")"
			done
		done
	done
done
