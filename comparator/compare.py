"""
comparator/compare.py

Result normalization and semantic comparison for RetORM.

The comparator must tolerate presentation-layer differences such as:
  - qualified vs unqualified column names
  - ORM label formatting differences
  - Decimal vs float
  - NaN vs None
  - unordered result rows when there is no ORDER BY
  - small float precision differences
"""

import math
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


Row = Dict[str, Any]
Rows = List[Row]
NormRow = Dict[str, Any]
NormRows = List[NormRow]


FLOAT_TOLERANCE = 1e-6
AGG_FLOAT_TOLERANCE = 1e-4


class CompareResult:
    """Comparison result with boolean success flag and diagnostic details."""

    def __init__(self, match: bool, reason: str = "", details: dict = None):
        self.match = match
        self.reason = reason
        self.details = details or {}

    def __bool__(self):
        return self.match

    def __repr__(self):
        if self.match:
            return "CompareResult(match=True)"
        return f"CompareResult(match=False, reason={self.reason!r})"


class AggFloat(float):
    """
    Marker type for floats that originate from DB aggregate results.

    DB-side aggregate values often lose precision relative to Python-side
    reference execution, so we compare them with a slightly larger tolerance.
    """


# Generated scan/join/derived aliases are short lowercase names like `u`, `u2`,
# `ps`, `dt`; preserve semantic aliases such as `avg_u_price` or `sum_total`.
_ALIAS_PREFIX_RE = re.compile(r"^([a-z]{1,2}\d*)_(.+)$")
_DUP_SUFFIX_RE = re.compile(r"__dup\d+$")


def compare_all(
    ref_rows: Rows,
    sql_rows: Rows,
    orm_rows: Rows,
    ordered: bool = False,
    strict: bool = False,
) -> Tuple[CompareResult, CompareResult]:
    norm_ref = normalize(ref_rows, strict=strict)
    norm_sql = normalize(sql_rows, strict=strict)
    norm_orm = normalize(orm_rows, strict=strict)

    ref_vs_sql = _compare_two(norm_ref, norm_sql, ordered, "ref", "sql", strict=strict)
    ref_vs_orm = _compare_two(norm_ref, norm_orm, ordered, "ref", "orm", strict=strict)
    return ref_vs_sql, ref_vs_orm


def compare_two_paths(
    rows_a: Rows,
    rows_b: Rows,
    name_a: str = "A",
    name_b: str = "B",
    ordered: bool = False,
    strict: bool = False,
) -> CompareResult:
    norm_a = normalize(rows_a, strict=strict)
    norm_b = normalize(rows_b, strict=strict)
    return _compare_two(norm_a, norm_b, ordered, name_a, name_b, strict=strict)


def normalize(rows: Rows, strict: bool = False) -> NormRows:
    return [_normalize_row(row, strict=strict) for row in rows]


def _normalize_row(row: Row, strict: bool = False) -> NormRow:
    norm_keys_in_order: List[str] = []
    norm_key_count: Dict[str, int] = {}
    for key in row.keys():
        norm_key = _normalize_key(key, strict=strict)
        norm_keys_in_order.append(norm_key)
        norm_key_count[norm_key] = norm_key_count.get(norm_key, 0) + 1

    result: NormRow = {}
    seen_count: Dict[str, int] = {}
    for (key, value), norm_key in zip(row.items(), norm_keys_in_order):
        norm_value = _normalize_value(value)
        if norm_key_count[norm_key] > 1:
            seen_count[norm_key] = seen_count.get(norm_key, 0) + 1
            canonical_key = f"{norm_key}#{seen_count[norm_key]}"
        else:
            canonical_key = norm_key
        result[canonical_key] = norm_value
    return result


def _normalize_key(key: str, strict: bool = False) -> str:
    if strict:
        return key

    key = _DUP_SUFFIX_RE.sub("", key)

    if "." in key:
        return key.split(".", 1)[1]

    if "_" in key:
        match = _ALIAS_PREFIX_RE.match(key)
        if match:
            return match.group(2)

    return key


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return AggFloat(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _compare_two(
    norm_a: NormRows,
    norm_b: NormRows,
    ordered: bool,
    name_a: str,
    name_b: str,
    strict: bool = False,
) -> CompareResult:
    if strict:
        ordered = True

    if len(norm_a) != len(norm_b):
        return CompareResult(
            match=False,
            reason=f"row count differs: {name_a}={len(norm_a)}, {name_b}={len(norm_b)}",
            details={name_a: norm_a, name_b: norm_b},
        )

    if not norm_a:
        return CompareResult(match=True, reason="both results are empty")

    keys_a = set(norm_a[0].keys())
    keys_b = set(norm_b[0].keys())
    if keys_a != keys_b:
        return CompareResult(
            match=False,
            reason=f"column set differs: {name_a}={sorted(keys_a)}, {name_b}={sorted(keys_b)}",
            details={"keys_a": sorted(keys_a), "keys_b": sorted(keys_b)},
        )

    if not ordered:
        return _compare_unordered_rows(norm_a, norm_b, name_a, name_b)

    for row_index, (row_a, row_b) in enumerate(zip(norm_a, norm_b)):
        mismatch = _first_row_mismatch(row_a, row_b)
        if mismatch is None:
            continue
        column, value_a, value_b = mismatch
        return CompareResult(
            match=False,
            reason=(
                f"row {row_index + 1}, column {column!r} differs: "
                f"{name_a}={value_a!r}, {name_b}={value_b!r}"
            ),
            details={
                "row_index": row_index,
                "column": column,
                name_a: row_a,
                name_b: row_b,
            },
        )

    return CompareResult(match=True)


def _compare_unordered_rows(
    norm_a: NormRows,
    norm_b: NormRows,
    name_a: str,
    name_b: str,
) -> CompareResult:
    """
    Compare unordered rows as multisets under semantic equality.

    This avoids false positives where approximate-equal float rows sort into
    different orders and then get compared against the wrong partner row.
    """

    adjacency: List[List[int]] = []
    for row_a in norm_a:
        adjacency.append(
            [b_idx for b_idx, row_b in enumerate(norm_b) if _rows_equal(row_a, row_b)]
        )

    match_to_a = [-1] * len(norm_b)

    def augment(a_idx: int, seen_b: List[bool]) -> bool:
        for b_idx in adjacency[a_idx]:
            if seen_b[b_idx]:
                continue
            seen_b[b_idx] = True
            if match_to_a[b_idx] == -1 or augment(match_to_a[b_idx], seen_b):
                match_to_a[b_idx] = a_idx
                return True
        return False

    matched = 0
    for a_idx in range(len(norm_a)):
        if augment(a_idx, [False] * len(norm_b)):
            matched += 1

    if matched == len(norm_a):
        return CompareResult(match=True)

    matched_a = {a_idx for a_idx in match_to_a if a_idx != -1}
    unmatched_a_idx = next(
        (idx for idx in range(len(norm_a)) if idx not in matched_a),
        0,
    )
    row_a = norm_a[unmatched_a_idx]
    remaining_b_indices = [idx for idx, a_idx in enumerate(match_to_a) if a_idx == -1]
    closest_b_idx = _find_closest_row_index(row_a, norm_b, remaining_b_indices)
    row_b = norm_b[closest_b_idx] if closest_b_idx is not None else {}

    mismatch = _first_row_mismatch(row_a, row_b) if row_b else None
    if mismatch is None:
        reason = f"unordered result sets differ: {name_b} has no semantic match for a {name_a} row"
        column = None
    else:
        column, value_a, value_b = mismatch
        reason = (
            f"unordered result sets differ: no {name_b} row semantically matches "
            f"a {name_a} row; first mismatch at column {column!r}: "
            f"{name_a}={value_a!r}, {name_b}={value_b!r}"
        )

    return CompareResult(
        match=False,
        reason=reason,
        details={
            "row_index": unmatched_a_idx,
            "column": column,
            name_a: row_a,
            name_b: row_b,
        },
    )


def _rows_equal(row_a: NormRow, row_b: NormRow) -> bool:
    if row_a.keys() != row_b.keys():
        return False
    for column in row_a.keys():
        if not _values_equal(row_a[column], row_b[column]):
            return False
    return True


def _first_row_mismatch(
    row_a: NormRow,
    row_b: NormRow,
) -> Optional[Tuple[str, Any, Any]]:
    if row_a.keys() != row_b.keys():
        return ("<keys>", sorted(row_a.keys()), sorted(row_b.keys()))
    for column in sorted(row_a.keys()):
        value_a = row_a[column]
        value_b = row_b.get(column)
        if not _values_equal(value_a, value_b):
            return (column, value_a, value_b)
    return None


def _find_closest_row_index(
    row_a: NormRow,
    candidate_rows: NormRows,
    candidate_indices: List[int],
) -> Optional[int]:
    if not candidate_indices:
        return None

    best_idx = None
    best_score = None
    for idx in candidate_indices:
        score = _row_distance(row_a, candidate_rows[idx])
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _row_distance(row_a: NormRow, row_b: NormRow) -> Tuple[int, int]:
    mismatch = _first_row_mismatch(row_a, row_b)
    if mismatch is None:
        return (0, 0)
    if mismatch[0] == "<keys>":
        return (10**9, 10**9)

    mismatch_count = 0
    numeric_mismatch = 0
    for column in row_a.keys():
        if _values_equal(row_a[column], row_b[column]):
            continue
        mismatch_count += 1
        if isinstance(row_a[column], (int, float)) and isinstance(row_b[column], (int, float)):
            numeric_mismatch += 1
    return (mismatch_count, numeric_mismatch)


def _sort_rows(rows: NormRows, sort_keys: List[str]) -> NormRows:
    """
    Keep the old helper for debugging/manual use.

    Ordered comparison no longer depends on this helper for unordered result
    alignment, but other callers may still find it useful.
    """

    def sort_key(row: NormRow):
        result = []
        for key in sort_keys:
            value = row.get(key)
            if value is None:
                result.append((0, ""))
            elif isinstance(value, str):
                result.append((1, value))
            else:
                result.append((1, value))
        return tuple(result)

    return sorted(rows, key=sort_key)


def _values_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if isinstance(a, bool) or isinstance(b, bool):
        return a == b

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if isinstance(a, AggFloat) or isinstance(b, AggFloat):
            return math.isclose(
                fa,
                fb,
                rel_tol=AGG_FLOAT_TOLERANCE,
                abs_tol=AGG_FLOAT_TOLERANCE,
            )
        return math.isclose(
            fa,
            fb,
            rel_tol=FLOAT_TOLERANCE,
            abs_tol=FLOAT_TOLERANCE,
        )

    return a == b


def print_report(
    ref_vs_sql: CompareResult,
    ref_vs_orm: CompareResult,
    ir_desc: str = "",
) -> None:
    print("\n" + "=" * 60)
    if ir_desc:
        print(f"IR: {ir_desc}")

    print(f"[ref vs sql] {'MATCH' if ref_vs_sql.match else 'MISMATCH'}")
    if not ref_vs_sql.match:
        print(f"  reason: {ref_vs_sql.reason}")
        _print_details(ref_vs_sql.details)

    print(f"[ref vs orm] {'MATCH' if ref_vs_orm.match else 'MISMATCH'}")
    if not ref_vs_orm.match:
        print(f"  reason: {ref_vs_orm.reason}")
        _print_details(ref_vs_orm.details)

    if ref_vs_sql.match and ref_vs_orm.match:
        print("=> all three paths are consistent")
    elif not ref_vs_sql.match:
        print("=> investigate SQL translation or reference semantics first")
    else:
        print("=> investigate ORM translation/assembly first")
    print("=" * 60)


def _print_details(details: dict) -> None:
    for key, value in details.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"    {key}: ({len(value)} rows, first 5 shown) {value[:5]}")
        else:
            print(f"    {key}: {value}")
