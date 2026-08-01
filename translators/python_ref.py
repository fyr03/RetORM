"""
translators/python_ref.py

Path 1: reference execution in Python.
"""

import math
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, __file__.rsplit("/translators", 1)[0])

from ir.nodes import (
    AggFunc,
    Aggregate,
    And,
    ArithExpr,
    ArithOp,
    Between,
    CaseWhen,
    CmpOp,
    Compare,
    Condition,
    DerivedTable,
    Distinct,
    Exists,
    Filter,
    GroupBy,
    Having,
    InList,
    InSubquery,
    Join,
    JoinType,
    Like,
    LimitOffset,
    Not,
    Or,
    OrderBy,
    Project,
    QueryNode,
    ScalarSubquery,
    Scan,
    SelectItem,
    SetOp,
    SetQuery,
    WindowExpr,
    WindowFunc,
)


Row = Dict[str, Any]
Rows = List[Row]
_UNRESOLVED = object()


class _ProjectedRow(dict):
    """Projected output row that retains its child row for later ORDER BY use."""

    def __init__(self, *args, source_row: Optional[Row] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_row = source_row


def execute(ir: QueryNode) -> Rows:
    return _eval(ir, outer_row=None)


def _eval(node: QueryNode, outer_row: Optional[Row] = None) -> Rows:
    if isinstance(node, Scan):
        return _eval_scan(node)
    if isinstance(node, DerivedTable):
        return _eval_derived_table(node, outer_row)
    if isinstance(node, Filter):
        return _eval_filter(node, outer_row)
    if isinstance(node, Join):
        return _eval_join(node, outer_row)
    if isinstance(node, GroupBy):
        return _eval_groupby(node, outer_row)
    if isinstance(node, Having):
        return _eval_having(node, outer_row)
    if isinstance(node, Project):
        return _eval_project(node, outer_row)
    if isinstance(node, Distinct):
        return _eval_distinct(node, outer_row)
    if isinstance(node, OrderBy):
        return _eval_orderby(node, outer_row)
    if isinstance(node, LimitOffset):
        return _eval_limit_offset(node, outer_row)
    if isinstance(node, SetQuery):
        return _eval_set_query(node, outer_row)
    raise NotImplementedError(f"unknown node type: {type(node)}")


def _eval_scan(node: Scan) -> Rows:
    from db.connector import execute_sql

    raw_rows = execute_sql(f"SELECT * FROM `{node.table}`;")
    return [{f"{node.alias}.{col}": val for col, val in row.items()} for row in raw_rows]


def _eval_derived_table(node: DerivedTable, outer_row: Optional[Row]) -> Rows:
    raw_rows = _eval(node.subquery, outer_row=outer_row)
    result = []
    for row in raw_rows:
        out_row = {}
        for key, value in row.items():
            out_row[f"{node.alias}.{_derived_output_name(key)}"] = value
        result.append(out_row)
    return result


def _eval_filter(node: Filter, outer_row: Optional[Row]) -> Rows:
    rows = _eval(node.child, outer_row=outer_row)
    return [row for row in rows if _eval_condition(node.condition, row, outer_row=outer_row)]


def _eval_join(node: Join, outer_row: Optional[Row]) -> Rows:
    left_rows = _eval(node.left, outer_row=outer_row)
    right_rows = _eval(node.right, outer_row=outer_row)
    right_null_row = {key: None for key in right_rows[0].keys()} if right_rows else {}

    result = []
    for left_row in left_rows:
        matched = False
        for right_row in right_rows:
            merged = {**left_row, **right_row}
            if _eval_condition(node.on, merged, outer_row=outer_row):
                matched = True
                result.append(merged)
        if node.join_type == JoinType.LEFT and not matched:
            result.append({**left_row, **right_null_row})
    return result


def _eval_groupby(node: GroupBy, outer_row: Optional[Row]) -> Rows:
    rows = _eval(node.child, outer_row=outer_row)
    groups: Dict[tuple, Rows] = defaultdict(list)
    for row in rows:
        group_key = tuple(
            _normalize_group_atom(_eval_expr(field, row, prefer_field=True, outer_row=outer_row))
            for field in node.fields
        )
        groups[group_key].append(row)

    result = []
    for group_key, group_rows in groups.items():
        out_row: Row = {}
        for field, value in zip(node.fields, group_key):
            if isinstance(field, str):
                out_row[field] = value
            else:
                out_row[repr(field)] = value
        for agg in node.aggregates:
            out_row[agg.alias] = _compute_aggregate(agg, group_rows, outer_row=outer_row)
        result.append(out_row)
    return result


def _compute_aggregate(agg: Aggregate, rows: Rows, outer_row: Optional[Row]) -> Any:
    if agg.func == AggFunc.COUNT:
        if agg.field == "*":
            return len(rows)
        return sum(
            1
            for row in rows
            if _eval_expr(agg.field, row, prefer_field=True, outer_row=outer_row) is not None
        )

    values = [
        _eval_expr(agg.field, row, prefer_field=True, outer_row=outer_row)
        for row in rows
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None

    if agg.func == AggFunc.SUM:
        return sum(values)
    if agg.func == AggFunc.AVG:
        return sum(values) / len(values)
    if agg.func == AggFunc.MAX:
        return max(values)
    if agg.func == AggFunc.MIN:
        return min(values)
    raise NotImplementedError(f"unknown aggregate function: {agg.func}")


def _eval_having(node: Having, outer_row: Optional[Row]) -> Rows:
    rows = _eval(node.child, outer_row=outer_row)
    return [row for row in rows if _eval_condition(node.condition, row, outer_row=outer_row)]


def _eval_project(node: Project, outer_row: Optional[Row]) -> Rows:
    rows = _eval(node.child, outer_row=outer_row)
    window_cache = _compute_window_caches(node.fields, rows, outer_row=outer_row)
    result = []
    for row in rows:
        out_row: Row = _ProjectedRow(source_row=_row_source(row))
        for field in node.fields:
            if isinstance(field, SelectItem):
                if isinstance(field.expr, WindowExpr):
                    out_row[field.alias] = window_cache[id(field.expr)][id(row)]
                else:
                    out_row[field.alias] = _eval_expr(
                        field.expr,
                        row,
                        prefer_field=True,
                        outer_row=outer_row,
                    )
            else:
                out_row[field] = _eval_expr(field, row, prefer_field=True, outer_row=outer_row)
        result.append(out_row)
    return result


def _eval_distinct(node: Distinct, outer_row: Optional[Row]) -> Rows:
    rows = _eval(node.child, outer_row=outer_row)
    seen = set()
    result = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _eval_orderby(node: OrderBy, outer_row: Optional[Row]) -> Rows:
    rows = list(_eval(node.child, outer_row=outer_row))
    window_order_cache = _compute_orderby_window_caches(
        node.keys,
        rows,
        outer_row=outer_row,
    )
    for key in reversed(node.keys):
        rows.sort(
            key=lambda row: _sort_atom(
                window_order_cache[id(key.field)][id(row)]
                if isinstance(key.field, WindowExpr)
                else _eval_expr(key.field, row, prefer_field=True, outer_row=outer_row)
            ),
            reverse=key.descending,
        )
    return rows


def _eval_limit_offset(node: LimitOffset, outer_row: Optional[Row]) -> Rows:
    rows = _eval(node.child, outer_row=outer_row)
    start = max(0, node.offset)
    end = start + max(0, node.limit)
    return rows[start:end]


def _eval_set_query(node: SetQuery, outer_row: Optional[Row]) -> Rows:
    left_rows = _eval(node.left, outer_row=outer_row)
    right_rows = _eval(node.right, outer_row=outer_row)

    if node.op == SetOp.UNION:
        if node.all:
            return left_rows + right_rows
        seen = set()
        result = []
        for row in left_rows + right_rows:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
        return result

    left_counter = Counter(_row_key(row) for row in left_rows)
    right_counter = Counter(_row_key(row) for row in right_rows)
    exemplar = {}
    for row in left_rows + right_rows:
        exemplar.setdefault(_row_key(row), row)

    result = []
    if node.op == SetOp.INTERSECT:
        for key in left_counter.keys() & right_counter.keys():
            copies = min(left_counter[key], right_counter[key]) if node.all else 1
            result.extend([exemplar[key]] * copies)
        return result
    if node.op == SetOp.EXCEPT:
        for key, left_count in left_counter.items():
            if node.all:
                copies = left_count - right_counter.get(key, 0)
            else:
                copies = 1 if key not in right_counter else 0
            if copies > 0:
                result.extend([exemplar[key]] * copies)
        return result
    raise NotImplementedError(f"unknown set op: {node.op}")


def _sort_atom(value: Any):
    if isinstance(value, float) and math.isnan(value):
        value = None
    if value is None:
        return (0, None)
    return (1, value)


def _row_key(row: Row):
    return tuple(sorted((key, _normalize_hashable(value)) for key, value in row.items()))


def _normalize_hashable(value: Any):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _eval_condition(cond: Condition, row: Row, outer_row: Optional[Row] = None) -> bool:
    result = _eval_condition_3vl(cond, row, outer_row=outer_row)
    return result is True


def _eval_condition_3vl(cond: Condition, row: Row, outer_row: Optional[Row] = None):
    if isinstance(cond, Compare):
        left_val = _eval_expr(cond.field, row, prefer_field=True, outer_row=outer_row)
        right_val = _eval_expr(cond.value, row, prefer_field=True, outer_row=outer_row)

        if right_val is None and _expr_is_literal_none(cond.value):
            if cond.op == CmpOp.EQ:
                return left_val is None
            if cond.op == CmpOp.NEQ:
                return left_val is not None

        if left_val is None or right_val is None:
            return None
        return _compare(left_val, cond.op, right_val)

    if isinstance(cond, InList):
        left_val = _eval_expr(cond.field, row, prefer_field=True, outer_row=outer_row)
        if left_val is None:
            return None
        values = [
            _eval_expr(value, row, prefer_field=True, outer_row=outer_row)
            for value in cond.values
        ]
        if any(value is None for value in values):
            values = [value for value in values if value is not None]
        result = left_val in values
        return (not result) if cond.negated else result

    if isinstance(cond, Between):
        value = _eval_expr(cond.field, row, prefer_field=True, outer_row=outer_row)
        lower = _eval_expr(cond.lower, row, prefer_field=True, outer_row=outer_row)
        upper = _eval_expr(cond.upper, row, prefer_field=True, outer_row=outer_row)
        if value is None or lower is None or upper is None:
            return None
        result = lower <= value <= upper
        return (not result) if cond.negated else result

    if isinstance(cond, Like):
        value = _eval_expr(cond.field, row, prefer_field=True, outer_row=outer_row)
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        result = _like_match(value, cond.pattern)
        return (not result) if cond.negated else result

    if isinstance(cond, Exists):
        correlated_outer = {**(outer_row or {}), **row}
        result = bool(_eval(cond.subquery, outer_row=correlated_outer))
        return (not result) if cond.negated else result

    if isinstance(cond, InSubquery):
        left_val = _eval_expr(cond.field, row, prefer_field=True, outer_row=outer_row)
        if left_val is None:
            return None
        correlated_outer = {**(outer_row or {}), **row}
        sub_rows = _eval(cond.subquery, outer_row=correlated_outer)
        values = _extract_subquery_values(sub_rows)
        non_null_values = [value for value in values if value is not None]
        result = left_val in non_null_values
        if not result and any(value is None for value in values):
            return None
        return (not result) if cond.negated else result

    if isinstance(cond, And):
        left = _eval_condition_3vl(cond.left, row, outer_row=outer_row)
        right = _eval_condition_3vl(cond.right, row, outer_row=outer_row)
        if left is False or right is False:
            return False
        if left is None or right is None:
            return None
        return True

    if isinstance(cond, Or):
        left = _eval_condition_3vl(cond.left, row, outer_row=outer_row)
        right = _eval_condition_3vl(cond.right, row, outer_row=outer_row)
        if left is True or right is True:
            return True
        if left is None or right is None:
            return None
        return False

    if isinstance(cond, Not):
        child = _eval_condition_3vl(cond.child, row, outer_row=outer_row)
        if child is None:
            return None
        return not child

    raise NotImplementedError(f"unknown condition type: {type(cond)}")


def _eval_expr(expr, row: Row, prefer_field: bool = True, outer_row: Optional[Row] = None):
    if isinstance(expr, ArithExpr):
        left = _eval_expr(expr.left, row, prefer_field=True, outer_row=outer_row)
        right = _eval_expr(expr.right, row, prefer_field=True, outer_row=outer_row)
        if left is None or right is None:
            return None
        return _eval_arith(left, expr.op, right)

    if isinstance(expr, CaseWhen):
        for case in expr.cases:
            if _eval_condition(case.condition, row, outer_row=outer_row):
                return _eval_expr(case.value, row, prefer_field=True, outer_row=outer_row)
        return _eval_expr(expr.else_value, row, prefer_field=False, outer_row=outer_row)

    if isinstance(expr, ScalarSubquery):
        correlated_outer = {**(outer_row or {}), **row}
        sub_rows = _eval(expr.subquery, outer_row=correlated_outer)
        if not sub_rows:
            return None
        if len(sub_rows) > 1:
            raise ValueError("scalar subquery returned more than one row")
        first_row = sub_rows[0]
        if not first_row:
            return None
        first_key = next(iter(first_row))
        return first_row[first_key]

    if isinstance(expr, WindowExpr):
        raise ValueError("window expressions should be evaluated in project context")

    if isinstance(expr, str) and prefer_field:
        resolved = _resolve_field(expr, row, outer_row=outer_row)
        if resolved is _UNRESOLVED:
            source_row = getattr(row, "source_row", None)
            if source_row is not None and source_row is not row:
                resolved = _resolve_field(expr, source_row, outer_row=outer_row)
        if resolved is not _UNRESOLVED:
            return resolved
        if "." in expr:
            raise ValueError(f"unresolved field in reference path: {expr!r}")

    return expr


def _eval_arith(left: Any, op: ArithOp, right: Any):
    if op == ArithOp.ADD:
        return left + right
    if op == ArithOp.SUB:
        return left - right
    if op == ArithOp.MUL:
        return left * right
    if op == ArithOp.DIV:
        if right == 0:
            return None
        return left / right
    if op == ArithOp.MOD:
        if right == 0:
            return None
        return left % right
    raise NotImplementedError(f"unknown arithmetic op: {op}")


def _compare(left: Any, op: CmpOp, right: Any) -> bool:
    if left is None or right is None:
        return False
    if op == CmpOp.EQ:
        return left == right
    if op == CmpOp.NEQ:
        return left != right
    if op == CmpOp.GT:
        return left > right
    if op == CmpOp.GTE:
        return left >= right
    if op == CmpOp.LT:
        return left < right
    if op == CmpOp.LTE:
        return left <= right
    raise NotImplementedError(f"unknown compare op: {op}")


def _normalize_group_atom(value: Any):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _like_match(value: str, pattern: str) -> bool:
    regex = "^" + "".join(_translate_like_char(ch) for ch in pattern) + "$"
    return re.match(regex, value, flags=re.DOTALL) is not None


def _translate_like_char(ch: str) -> str:
    if ch == "%":
        return ".*"
    if ch == "_":
        return "."
    return re.escape(ch)


def _expr_is_literal_none(expr) -> bool:
    return expr is None


def _extract_subquery_values(rows: Rows) -> List[Any]:
    values: List[Any] = []
    for row in rows:
        if not row:
            values.append(None)
            continue
        first_key = next(iter(row))
        values.append(row[first_key])
    return values


def _resolve_field(field_name: str, row: Row, outer_row: Optional[Row] = None):
    if field_name in row:
        return row[field_name]
    if outer_row and field_name in outer_row:
        return outer_row[field_name]

    if "." not in field_name:
        matches = [value for key, value in row.items() if key.endswith(f".{field_name}")]
        if not matches and outer_row:
            matches = [value for key, value in outer_row.items() if key.endswith(f".{field_name}")]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(
                f"[python_ref] warning: field {field_name!r} is ambiguous, "
                f"matched keys={[key for key in row if key.endswith(f'.{field_name}')]}",
            )
            return matches[0]

    return _UNRESOLVED


def _row_source(row: Row) -> Row:
    source_row = getattr(row, "source_row", None)
    if source_row is None or source_row is row:
        return row
    return _row_source(source_row)


def _derived_output_name(field_name: str) -> str:
    return field_name.split(".", 1)[1] if "." in field_name else field_name


def _compute_window_caches(fields: List[object], rows: Rows, outer_row: Optional[Row] = None):
    caches = {}
    for field in fields:
        if isinstance(field, SelectItem) and isinstance(field.expr, WindowExpr):
            caches[id(field.expr)] = _eval_window_expr_for_rows(field.expr, rows, outer_row=outer_row)
    return caches


def _compute_orderby_window_caches(order_keys, rows: Rows, outer_row: Optional[Row] = None):
    caches = {}
    for order_key in order_keys:
        if isinstance(order_key.field, WindowExpr):
            caches[id(order_key.field)] = _eval_window_expr_for_rows(
                order_key.field,
                rows,
                outer_row=outer_row,
            )
    return caches


def _eval_window_expr_for_rows(expr: WindowExpr, rows: Rows, outer_row: Optional[Row] = None):
    partitions = defaultdict(list)
    for idx, row in enumerate(rows):
        key = tuple(
            _normalize_group_atom(_eval_expr(item, row, prefer_field=True, outer_row=outer_row))
            for item in expr.partition_by
        )
        partitions[key].append((idx, row))

    result = {}
    for part_rows in partitions.values():
        ordered = list(part_rows)
        for order_key in reversed(expr.order_by):
            ordered.sort(
                key=lambda item: _sort_atom(
                    _eval_expr(order_key.field, item[1], prefer_field=True, outer_row=outer_row)
                ),
                reverse=order_key.descending,
            )

        if expr.func == WindowFunc.ROW_NUMBER:
            for pos, (_, row) in enumerate(ordered, start=1):
                result[id(row)] = pos
            continue

        if expr.func in (WindowFunc.RANK, WindowFunc.DENSE_RANK):
            prev_order = _UNRESOLVED
            rank = 0
            dense_rank = 0
            for pos, (_, row) in enumerate(ordered, start=1):
                order_value = tuple(
                    _eval_expr(key.field, row, prefer_field=True, outer_row=outer_row)
                    for key in expr.order_by
                )
                if order_value != prev_order:
                    rank = pos
                    dense_rank += 1
                    prev_order = order_value
                result[id(row)] = rank if expr.func == WindowFunc.RANK else dense_rank
            continue

        values = []
        for _, row in ordered:
            if expr.field == "*":
                values.append(1)
            else:
                values.append(_eval_expr(expr.field, row, prefer_field=True, outer_row=outer_row))
        non_null = [value for value in values if value is not None]

        if expr.func == WindowFunc.COUNT:
            window_value = len(ordered) if expr.field == "*" else len(non_null)
        elif not non_null:
            window_value = None
        elif expr.func == WindowFunc.SUM:
            window_value = sum(non_null)
        elif expr.func == WindowFunc.AVG:
            window_value = sum(non_null) / len(non_null)
        elif expr.func == WindowFunc.MAX:
            window_value = max(non_null)
        elif expr.func == WindowFunc.MIN:
            window_value = min(non_null)
        else:
            raise NotImplementedError(f"unsupported window function: {expr.func}")

        for _, row in ordered:
            result[id(row)] = window_value

    return result
