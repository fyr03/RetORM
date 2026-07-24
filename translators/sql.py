"""
translators/sql.py

Path 2: IR -> Raw SQL.
"""

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connector import execute_sql
from ir.nodes import (
    Aggregate,
    And,
    CmpOp,
    Compare,
    Condition,
    Filter,
    GroupBy,
    Having,
    Join,
    JoinType,
    Not,
    Or,
    Project,
    QueryNode,
    Scan,
)


def execute(ir: QueryNode) -> List[dict]:
    sql = translate(ir)
    print(f"[sql] generated SQL:\n  {sql}")
    return execute_sql(sql)


def translate(ir: QueryNode) -> str:
    select_clause, from_clause, where_clause, groupby_clause, having_clause = _translate(ir)

    parts = [select_clause, from_clause]
    if where_clause:
        parts.append(f"WHERE {where_clause}")
    if groupby_clause:
        parts.append(f"GROUP BY {groupby_clause}")
    if having_clause:
        parts.append(f"HAVING {having_clause}")
    return "\n".join(parts) + ";"


def _translate(
    node: QueryNode,
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    if isinstance(node, Scan):
        return _translate_scan(node)
    if isinstance(node, Filter):
        return _translate_filter(node)
    if isinstance(node, Join):
        return _translate_join(node)
    if isinstance(node, GroupBy):
        return _translate_groupby(node)
    if isinstance(node, Having):
        return _translate_having(node)
    if isinstance(node, Project):
        return _translate_project(node)
    raise NotImplementedError(f"unknown node type: {type(node)}")


def _translate_scan(node: Scan):
    return "SELECT *", f"FROM `{node.table}` AS `{node.alias}`", None, None, None


def _translate_filter(node: Filter):
    select, from_, where, groupby, having = _translate(node.child)
    new_where = _translate_condition(node.condition)
    return (
        select,
        from_,
        f"({where}) AND ({new_where})" if where else new_where,
        groupby,
        having,
    )


def _translate_join(node: Join):
    on = _translate_condition(node.on)
    join_keyword = "LEFT JOIN" if node.join_type == JoinType.LEFT else "INNER JOIN"
    from_ = f"FROM {_render_from_expr(node.left)}\n{join_keyword} {_render_from_expr(node.right)} ON {on}"
    return "SELECT *", from_, None, None, None


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


def _translate_groupby(node: GroupBy):
    select, from_, where, _, having = _translate(node.child)
    group_cols = ", ".join(_quote_field(field) for field in node.fields)
    agg_exprs = [_translate_aggregate(agg) for agg in node.aggregates]
    select_cols = ", ".join([_quote_field(field) for field in node.fields] + agg_exprs)
    return f"SELECT {select_cols}", from_, where, group_cols, having


def _translate_aggregate(agg: Aggregate) -> str:
    if agg.field == "*":
        expr = f"{agg.func.value}(*)"
    else:
        expr = f"{agg.func.value}({_quote_field(agg.field)})"
    return f"{expr} AS `{agg.alias}`"


def _translate_having(node: Having):
    select, from_, where, groupby, _ = _translate(node.child)
    agg_expr_map = _collect_agg_exprs(node.child)
    having = _translate_condition(node.condition, agg_expr_map=agg_expr_map)
    return select, from_, where, groupby, having


def _collect_agg_exprs(node) -> dict:
    result = {}
    if isinstance(node, GroupBy):
        for agg in node.aggregates:
            if agg.field == "*":
                expr = f"{agg.func.value}(*)"
            else:
                expr = f"{agg.func.value}({_quote_field(agg.field)})"
            result[agg.alias] = expr
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Having):
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Filter):
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Join):
        result.update(_collect_agg_exprs(node.left))
        result.update(_collect_agg_exprs(node.right))
    return result


def _translate_project(node: Project):
    select, from_, where, groupby, having = _translate(node.child)
    agg_map = _collect_aggregates(node.child)
    col_exprs = []
    for field in node.fields:
        col_exprs.append(agg_map[field] if field in agg_map else _quote_field(field))
    return f"SELECT {', '.join(col_exprs)}", from_, where, groupby, having


def _collect_aggregates(node: QueryNode) -> dict:
    result = {}
    if isinstance(node, GroupBy):
        for agg in node.aggregates:
            result[agg.alias] = _translate_aggregate(agg)
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Having):
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Filter):
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Join):
        result.update(_collect_aggregates(node.left))
        result.update(_collect_aggregates(node.right))
    return result


def _translate_condition(cond: Condition, agg_expr_map: Optional[dict] = None) -> str:
    agg_expr_map = agg_expr_map or {}

    if isinstance(cond, Compare):
        left = agg_expr_map.get(cond.field, _quote_field(cond.field))
        if cond.value is None:
            if cond.op == CmpOp.EQ:
                return f"{left} IS NULL"
            if cond.op == CmpOp.NEQ:
                return f"{left} IS NOT NULL"
        return f"{left} {cond.op.value} {_quote_value(cond.value)}"

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


def _quote_field(field_name: str) -> str:
    if "." in field_name:
        alias, col = field_name.split(".", 1)
        return f"`{alias}`.`{col}`"
    return f"`{field_name}`"


def _quote_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str) and "." in value:
        return _quote_field(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{value}'"
