"""
translators/sql.py

Path 2: IR -> raw SQL.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connector import execute_sql
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
    Distinct,
    Filter,
    GroupBy,
    Having,
    InList,
    Join,
    JoinType,
    Like,
    LimitOffset,
    Not,
    Or,
    OrderBy,
    OrderKey,
    Project,
    QueryNode,
    Scan,
    SelectItem,
)


@dataclass
class QueryParts:
    select_clause: str
    from_clause: str
    where_clause: Optional[str] = None
    groupby_clause: Optional[str] = None
    having_clause: Optional[str] = None
    orderby_clause: Optional[str] = None
    limit_clause: Optional[str] = None


def execute(ir: QueryNode) -> List[dict]:
    sql = translate(ir)
    print(f"[sql] generated SQL:\n  {sql}")
    return execute_sql(sql)


def translate(ir: QueryNode) -> str:
    parts = _build_parts(ir)
    lines = [parts.select_clause, parts.from_clause]
    if parts.where_clause:
        lines.append(f"WHERE {parts.where_clause}")
    if parts.groupby_clause:
        lines.append(f"GROUP BY {parts.groupby_clause}")
    if parts.having_clause:
        lines.append(f"HAVING {parts.having_clause}")
    if parts.orderby_clause:
        lines.append(f"ORDER BY {parts.orderby_clause}")
    if parts.limit_clause:
        lines.append(parts.limit_clause)
    return "\n".join(lines) + ";"


def _build_parts(node: QueryNode) -> QueryParts:
    if isinstance(node, Scan):
        return QueryParts(
            select_clause="SELECT *",
            from_clause=f"FROM `{node.table}` AS `{node.alias}`",
        )
    if isinstance(node, Filter):
        parts = _build_parts(node.child)
        new_where = _translate_condition(node.condition, agg_expr_map=_collect_agg_exprs(node.child))
        parts.where_clause = (
            f"({parts.where_clause}) AND ({new_where})"
            if parts.where_clause
            else new_where
        )
        return parts
    if isinstance(node, Join):
        on = _translate_condition(node.on)
        join_keyword = "LEFT JOIN" if node.join_type == JoinType.LEFT else "INNER JOIN"
        from_clause = (
            f"FROM {_render_from_expr(node.left)}\n"
            f"{join_keyword} {_render_from_expr(node.right)} ON {on}"
        )
        return QueryParts(select_clause="SELECT *", from_clause=from_clause)
    if isinstance(node, GroupBy):
        parts = _build_parts(node.child)
        fields = [_translate_expr(field, prefer_field=True) for field in node.fields]
        agg_exprs = [_translate_aggregate(agg) for agg in node.aggregates]
        parts.select_clause = f"SELECT {', '.join(fields + agg_exprs)}"
        parts.groupby_clause = ", ".join(fields)
        return parts
    if isinstance(node, Having):
        parts = _build_parts(node.child)
        parts.having_clause = _translate_condition(
            node.condition,
            agg_expr_map=_collect_agg_exprs(node.child),
        )
        return parts
    if isinstance(node, Project):
        parts = _build_parts(node.child)
        agg_expr_map = _collect_agg_exprs(node.child)
        agg_select_map = _collect_aggregates(node.child)
        exprs = []
        for field in node.fields:
            if isinstance(field, SelectItem):
                exprs.append(
                    f"{_translate_expr(field.expr, agg_expr_map=agg_expr_map, prefer_field=True)} AS `{field.alias}`"
                )
            elif field in agg_select_map:
                exprs.append(agg_select_map[field])
            else:
                exprs.append(
                    _translate_expr(field, agg_expr_map=agg_expr_map, prefer_field=True)
                )
        parts.select_clause = f"SELECT {', '.join(exprs)}"
        return parts
    if isinstance(node, Distinct):
        parts = _build_parts(node.child)
        if parts.select_clause.startswith("SELECT DISTINCT "):
            return parts
        if parts.select_clause.startswith("SELECT "):
            parts.select_clause = "SELECT DISTINCT " + parts.select_clause[len("SELECT "):]
        return parts
    if isinstance(node, OrderBy):
        parts = _build_parts(node.child)
        agg_expr_map = _collect_agg_exprs(node.child)
        parts.orderby_clause = ", ".join(
            _translate_order_key(key, agg_expr_map=agg_expr_map)
            for key in node.keys
        )
        return parts
    if isinstance(node, LimitOffset):
        parts = _build_parts(node.child)
        limit = max(0, node.limit)
        offset = max(0, node.offset)
        parts.limit_clause = f"LIMIT {limit}" + (f" OFFSET {offset}" if offset else "")
        return parts
    raise NotImplementedError(f"unknown node type: {type(node)}")


def _render_from_expr(node: QueryNode) -> str:
    if isinstance(node, Scan):
        return f"`{node.table}` AS `{node.alias}`"
    if isinstance(node, Join):
        join_keyword = "LEFT JOIN" if node.join_type == JoinType.LEFT else "INNER JOIN"
        on = _translate_condition(node.on)
        return (
            f"({_render_from_expr(node.left)} "
            f"{join_keyword} {_render_from_expr(node.right)} ON {on})"
        )
    raise NotImplementedError(f"unsupported FROM node: {type(node)}")


def _translate_condition(cond: Condition, agg_expr_map: Optional[dict] = None) -> str:
    agg_expr_map = agg_expr_map or {}

    if isinstance(cond, Compare):
        left = _translate_expr(cond.field, agg_expr_map=agg_expr_map, prefer_field=True)
        if cond.value is None:
            if cond.op == CmpOp.EQ:
                return f"{left} IS NULL"
            if cond.op == CmpOp.NEQ:
                return f"{left} IS NOT NULL"
        right = _translate_expr(cond.value, agg_expr_map=agg_expr_map, prefer_field=True)
        return f"{left} {cond.op.value} {right}"

    if isinstance(cond, InList):
        left = _translate_expr(cond.field, agg_expr_map=agg_expr_map, prefer_field=True)
        values = ", ".join(
            _translate_expr(value, agg_expr_map=agg_expr_map, prefer_field=True)
            for value in cond.values
        )
        not_prefix = "NOT " if cond.negated else ""
        return f"{left} {not_prefix}IN ({values})"

    if isinstance(cond, Between):
        left = _translate_expr(cond.field, agg_expr_map=agg_expr_map, prefer_field=True)
        lower = _translate_expr(cond.lower, agg_expr_map=agg_expr_map, prefer_field=True)
        upper = _translate_expr(cond.upper, agg_expr_map=agg_expr_map, prefer_field=True)
        not_prefix = "NOT " if cond.negated else ""
        return f"{left} {not_prefix}BETWEEN {lower} AND {upper}"

    if isinstance(cond, Like):
        left = _translate_expr(cond.field, agg_expr_map=agg_expr_map, prefer_field=True)
        not_prefix = "NOT " if cond.negated else ""
        return f"{left} {not_prefix}LIKE {_quote_value(cond.pattern)}"

    if isinstance(cond, And):
        return (
            f"({_translate_condition(cond.left, agg_expr_map)}) AND "
            f"({_translate_condition(cond.right, agg_expr_map)})"
        )
    if isinstance(cond, Or):
        return (
            f"({_translate_condition(cond.left, agg_expr_map)}) OR "
            f"({_translate_condition(cond.right, agg_expr_map)})"
        )
    if isinstance(cond, Not):
        return f"NOT ({_translate_condition(cond.child, agg_expr_map)})"

    raise NotImplementedError(f"unknown condition type: {type(cond)}")


def _translate_expr(expr, agg_expr_map: Optional[dict] = None, prefer_field: bool = False) -> str:
    agg_expr_map = agg_expr_map or {}

    if isinstance(expr, ArithExpr):
        left = _translate_expr(expr.left, agg_expr_map=agg_expr_map, prefer_field=True)
        right = _translate_expr(expr.right, agg_expr_map=agg_expr_map, prefer_field=True)
        return f"({left} {expr.op.value} {right})"

    if isinstance(expr, CaseWhen):
        parts = []
        for case in expr.cases:
            parts.append(
                f"WHEN {_translate_condition(case.condition, agg_expr_map)} "
                f"THEN {_translate_expr(case.value, agg_expr_map=agg_expr_map, prefer_field=True)}"
            )
        else_sql = _translate_expr(expr.else_value, agg_expr_map=agg_expr_map, prefer_field=False)
        return f"(CASE {' '.join(parts)} ELSE {else_sql} END)"

    if isinstance(expr, str):
        if expr in agg_expr_map:
            return agg_expr_map[expr]
        if prefer_field and "." in expr:
            return _quote_field(expr)
        return _quote_value(expr)

    return _quote_value(expr)


def _translate_order_key(key: OrderKey, agg_expr_map: Optional[dict] = None) -> str:
    field_expr = _translate_expr(key.field, agg_expr_map=agg_expr_map or {}, prefer_field=True)
    suffix = "DESC" if key.descending else "ASC"
    return f"{field_expr} {suffix}"


def _translate_aggregate(agg: Aggregate) -> str:
    if agg.field == "*":
        expr = f"{agg.func.value}(*)"
    else:
        expr = f"{agg.func.value}({_translate_expr(agg.field, prefer_field=True)})"
    return f"{expr} AS `{agg.alias}`"


def _collect_agg_exprs(node: QueryNode) -> dict:
    result = {}
    if isinstance(node, GroupBy):
        for agg in node.aggregates:
            if agg.field == "*":
                expr = f"{agg.func.value}(*)"
            else:
                expr = f"{agg.func.value}({_translate_expr(agg.field, prefer_field=True)})"
            result[agg.alias] = expr
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, (Having, Filter, Project, Distinct, OrderBy, LimitOffset)):
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Join):
        result.update(_collect_agg_exprs(node.left))
        result.update(_collect_agg_exprs(node.right))
    return result


def _collect_aggregates(node: QueryNode) -> dict:
    result = {}
    if isinstance(node, GroupBy):
        for agg in node.aggregates:
            result[agg.alias] = _translate_aggregate(agg)
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, (Having, Filter, Project, Distinct, OrderBy, LimitOffset)):
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Join):
        result.update(_collect_aggregates(node.left))
        result.update(_collect_aggregates(node.right))
    return result


def _quote_field(field_name: str) -> str:
    if "." in field_name:
        alias, col = field_name.split(".", 1)
        return f"`{alias}`.`{col}`"
    return f"`{field_name}`"


def _quote_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "\\'") + "'"
