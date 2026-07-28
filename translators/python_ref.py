"""
translators/python_ref.py

Path 1: reference execution in Python.
"""

import math
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, __file__.rsplit("/translators", 1)[0])

from db.connector import execute_sql
from ir.nodes import (
    AggFunc,
    Aggregate,
    And,
    Compare,
    Condition,
    Distinct,
    Filter,
    GroupBy,
    Having,
    Join,
    JoinType,
    Not,
    Or,
    OrderBy,
    OrderKey,
    Project,
    QueryNode,
    Scan,
    CmpOp,
)


Row = Dict[str, Any]
Rows = List[Row]


def execute(ir: QueryNode) -> Rows:
    return _eval(ir)


def _eval(node: QueryNode) -> Rows:
    if isinstance(node, Scan):
        return _eval_scan(node)
    if isinstance(node, Filter):
        return _eval_filter(node)
    if isinstance(node, Join):
        return _eval_join(node)
    if isinstance(node, GroupBy):
        return _eval_groupby(node)
    if isinstance(node, Having):
        return _eval_having(node)
    if isinstance(node, Project):
        return _eval_project(node)
    if isinstance(node, Distinct):
        return _eval_distinct(node)
    if isinstance(node, OrderBy):
        return _eval_orderby(node)
    raise NotImplementedError(f"unknown node type: {type(node)}")


def _eval_scan(node: Scan) -> Rows:
    raw_rows = execute_sql(f"SELECT * FROM `{node.table}`;")
    result = []
    for row in raw_rows:
        result.append({f"{node.alias}.{col}": val for col, val in row.items()})
    return result


def _eval_filter(node: Filter) -> Rows:
    rows = _eval(node.child)
    return [row for row in rows if _eval_condition(node.condition, row)]


def _eval_join(node: Join) -> Rows:
    left_rows = _eval(node.left)
    right_rows = _eval(node.right)
    right_null_row = {key: None for key in right_rows[0].keys()} if right_rows else {}

    result = []
    for left_row in left_rows:
        matched = False
        for right_row in right_rows:
            merged = {**left_row, **right_row}
            if _eval_condition(node.on, merged):
                matched = True
                result.append(merged)
        if node.join_type == JoinType.LEFT and not matched:
            result.append({**left_row, **right_null_row})
    return result


def _eval_groupby(node: GroupBy) -> Rows:
    rows = _eval(node.child)
    groups: Dict[tuple, Rows] = defaultdict(list)
    for row in rows:
        group_key = tuple(
            None if (isinstance(v, float) and v != v) else v
            for v in (_resolve_field(field, row) for field in node.fields)
        )
        groups[group_key].append(row)

    result = []
    for group_key, group_rows in groups.items():
        out_row: Row = {}
        for field_name, value in zip(node.fields, group_key):
            out_row[field_name] = value
        for agg in node.aggregates:
            out_row[agg.alias] = _compute_aggregate(agg, group_rows)
        result.append(out_row)
    return result


def _compute_aggregate(agg: Aggregate, rows: Rows) -> Any:
    if agg.func == AggFunc.COUNT:
        if agg.field == "*":
            return len(rows)
        return sum(1 for row in rows if _resolve_field(agg.field, row) is not None)

    values = [_resolve_field(agg.field, row) for row in rows]
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


def _eval_having(node: Having) -> Rows:
    rows = _eval(node.child)
    return [row for row in rows if _eval_condition(node.condition, row)]


def _eval_project(node: Project) -> Rows:
    rows = _eval(node.child)
    result = []
    for row in rows:
        result.append({field_name: _resolve_field(field_name, row) for field_name in node.fields})
    return result


def _eval_distinct(node: Distinct) -> Rows:
    rows = _eval(node.child)
    seen = set()
    result = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _eval_orderby(node: OrderBy) -> Rows:
    rows = list(_eval(node.child))
    # Use stable multi-pass sorting so mixed ASC/DESC keys behave like SQL.
    for key in reversed(node.keys):
        rows.sort(
            key=lambda row: _sort_atom(_resolve_field(key.field, row)),
            reverse=key.descending,
        )
    return rows


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


def _eval_condition(cond: Condition, row: Row) -> bool:
    result = _eval_condition_3vl(cond, row)
    return result is True


def _eval_condition_3vl(cond: Condition, row: Row):
    if isinstance(cond, Compare):
        left_val = _resolve_field(cond.field, row)
        if isinstance(cond.value, str) and "." in cond.value:
            right_val = _resolve_field(cond.value, row)
        else:
            right_val = cond.value

        if right_val is None and not (isinstance(cond.value, str) and "." in str(cond.value)):
            if cond.op == CmpOp.EQ:
                return left_val is None
            if cond.op == CmpOp.NEQ:
                return left_val is not None

        if left_val is None or right_val is None:
            return None
        return _compare(left_val, cond.op, right_val)

    if isinstance(cond, And):
        left = _eval_condition_3vl(cond.left, row)
        right = _eval_condition_3vl(cond.right, row)
        if left is False or right is False:
            return False
        if left is None or right is None:
            return None
        return True

    if isinstance(cond, Or):
        left = _eval_condition_3vl(cond.left, row)
        right = _eval_condition_3vl(cond.right, row)
        if left is True or right is True:
            return True
        if left is None or right is None:
            return None
        return False

    if isinstance(cond, Not):
        child = _eval_condition_3vl(cond.child, row)
        if child is None:
            return None
        return not child

    raise NotImplementedError(f"unknown condition type: {type(cond)}")


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


def _resolve_field(field_name: str, row: Row) -> Any:
    if field_name in row:
        return row[field_name]

    if "." not in field_name:
        matches = [value for key, value in row.items() if key.endswith(f".{field_name}")]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(
                f"[python_ref] warning: field {field_name!r} is ambiguous, "
                f"matched keys={[key for key in row if key.endswith(f'.{field_name}')]}",
            )
            return matches[0]

    return None
