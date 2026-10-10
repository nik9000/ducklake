//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_multisequence_select.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

//! Multisequence selection: finds how many rows of each of several sorted sequences come before the rank-th row of
//! their merge, where rows order by key and then by sequence, like a stable k-way merge. See section 4.1 of Axtmann et
//! al., "Practical Massively Parallel Sorting" (https://arxiv.org/html/1410.6754v2), which pivots on a random row.
//! Here the pivot is the weighted median of the middles of the ranges still open, so every step settles at least a
//! quarter of the open rows: O(log n) steps of O(k log k + k log n) comparisons for k sequences of n rows.
//!
//! SEQUENCES provides Count() and Less(lhs_sequence, lhs_position, rhs_sequence, rhs_position), which orders rows of
//! different sequences by key and then by sequence.
template <class SEQUENCES>
struct MultisequenceSelection {
	//! Sets cuts[i] to the end of sequence i's rows among the first rank rows of the merge of rows [starts[i], ends[i])
	//! of every sequence, taking every row when there are fewer. Returns the number of steps taken.
	static idx_t Select(const SEQUENCES &sequences, const vector<idx_t> &starts, const vector<idx_t> &ends, idx_t rank,
	                    vector<idx_t> &cuts) {
		auto count = sequences.Count();
		// rows before lo[i] are among the first rank rows, rows from hi[i] on are not, the rows between are open
		vector<idx_t> lo(starts);
		vector<idx_t> hi(count);
		vector<idx_t> positions(count);
		for (idx_t i = 0; i < count; i++) {
			hi[i] = MinValue(ends[i], starts[i] + rank);
		}
		vector<Middle> middles;
		idx_t steps = 0;
		while (true) {
			idx_t open = 0;
			middles.clear();
			for (idx_t i = 0; i < count; i++) {
				if (hi[i] > lo[i]) {
					open += hi[i] - lo[i];
					middles.push_back(Middle {i, lo[i] + (hi[i] - lo[i]) / 2, hi[i] - lo[i]});
				}
			}
			if (open == 0) {
				break;
			}
			steps++;
			std::sort(middles.begin(), middles.end(), [&](const Middle &lhs, const Middle &rhs) {
				return sequences.Less(lhs.sequence, lhs.position, rhs.sequence, rhs.position);
			});
			// the weighted median of the middles: open rows weigh at least half of the open rows on either side of it
			idx_t weight = 0;
			auto pivot = middles.back();
			for (auto &middle : middles) {
				weight += middle.weight;
				if (2 * weight >= open) {
					pivot = middle;
					break;
				}
			}
			// the rows that come before the pivot
			idx_t before = 0;
			for (idx_t i = 0; i < count; i++) {
				if (i == pivot.sequence) {
					positions[i] = pivot.position;
				} else {
					positions[i] = FirstAfter(sequences, i, lo[i], hi[i], pivot);
				}
				before += positions[i] - starts[i];
			}
			if (before >= rank) {
				// the pivot and the rows after it are not among the first rank rows
				hi = positions;
			} else {
				// the pivot and the rows before it are
				lo = positions;
				lo[pivot.sequence]++;
			}
		}
		cuts = std::move(lo);
		return steps;
	}

private:
	struct Middle {
		idx_t sequence;
		idx_t position;
		//! The number of open rows of the sequence
		idx_t weight;
	};

	//! The first position in [lo, hi) of the sequence whose row comes after the pivot, hi if none does
	static idx_t FirstAfter(const SEQUENCES &sequences, idx_t sequence, idx_t lo, idx_t hi, const Middle &pivot) {
		while (lo < hi) {
			auto middle = lo + (hi - lo) / 2;
			if (sequences.Less(sequence, middle, pivot.sequence, pivot.position)) {
				lo = middle + 1;
			} else {
				hi = middle;
			}
		}
		return lo;
	}
};

} // namespace duckdb
