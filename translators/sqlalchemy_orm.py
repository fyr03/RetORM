"""
translators/sqlalchemy_orm.py

Path 3: IR -> SQLAlchemy Core.
"""

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import MetaData, and_, func, not_, or_, select
from sqlalchemy.engine import Engine

from db.connector import get_engine
from ir.nodes import (
    AggFunc,
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

Row = Dict[str, Any]
Rows = List[Row]

_metadata: Optional[MetaData] = None


def _get_metadata(engine: Engine) -> MetaData:
    global _metadata
    if _metadata is None:
        _metadata = MetaData()
        _metadata.reflect(bind=engine)
    return _metadata


def reset_metadata() -> None:
    global _metadata
    _metadata = None


def _get_table(table_name: str, engine: Engine):
    metadata = _get_metadata(engine)
    if table_name not in metadata.tables:
        raise KeyError(
            f"[sqlalchemy_orm] table '{table_name}' not reflected, "
            f"known tables: {list(metadata.tables.keys())}"
        )
    return metadata.tables[table_name]


def execute(ir: QueryNode) -> Rows:
    engine = get_engine()
    stmt, _ = _build_query(ir, engine)
    print(f"[sqlalchemy_orm] generated SQL:\n  {stmt}")

    rows: Rows = []
    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result:
            rows.append(dict(row._mapping))
    return rows


def _build_query(node: QueryNode, engine: Engine):
    ctx = _collect_ctx(node, engine)
    stmt = _assemble(ctx)
    return stmt, ctx


def _collect_ctx(node: QueryNode, engine: Engine) -> dict:
    if isinstance(node, Scan):
        return _build_scan_ctx(node, engine)
    if isinstance(node, Filter):
        return _build_filter_ctx(node, engine)
    if isinstance(node, Join):
        return _build_join_ctx(node, engine)
    if isinstance(node, GroupBy):
        return _build_groupby_ctx(node, engine)
    if isinstance(node, Having):
        return _build_having_ctx(node, engine)
    if isinstance(node, Project):
        return _build_project_ctx(node, engine)
    raise NotImplementedError(f"unknown node type: {type(node)}")


def _build_scan_ctx(node: Scan, engine: Engine) -> dict:
    table_alias = _get_table(node.table, engine).alias(node.alias)
    return {
        "froms": [table_alias],
        "tables": {node.alias: table_alias},
        "where": [],
        "groupby": [],
        "having": [],
        "agg_map": {},
        "select_cols": [table_alias],
    }


def _build_filter_ctx(node: Filter, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    ctx["where"].append(_build_condition(node.condition, ctx))
    return ctx


def _build_join_ctx(node: Join, engine: Engine) -> dict:
    left_ctx = _collect_ctx(node.left, engine)
    right_ctx = _collect_ctx(node.right, engine)

    ctx = {
        "froms": [],
        "tables": {**left_ctx["tables"], **right_ctx["tables"]},
        "where": left_ctx["where"] + right_ctx["where"],
        "groupby": left_ctx["groupby"] + right_ctx["groupby"],
        "having": left_ctx["having"] + right_ctx["having"],
        "agg_map": {**left_ctx["agg_map"], **right_ctx["agg_map"]},
        "select_cols": list(left_ctx["select_cols"]) + list(right_ctx["select_cols"]),
    }

    left_from = left_ctx["froms"][0]
    right_from = right_ctx["froms"][0]
    on_cond = _build_condition(node.on, ctx)
    joined = (
        left_from.outerjoin(right_from, on_cond)
        if node.join_type == JoinType.LEFT
        else left_from.join(right_from, on_cond)
    )
    ctx["froms"] = [joined]
    return ctx


def _build_groupby_ctx(node: GroupBy, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    groupby_cols = [_resolve_col(field, ctx) for field in node.fields]
    ctx["groupby"] = groupby_cols

    agg_exprs = []
    for agg in node.aggregates:
        expr = _build_aggregate(agg, ctx).label(agg.alias)
        ctx["agg_map"][agg.alias] = expr
        agg_exprs.append(expr)

    ctx["select_cols"] = groupby_cols + agg_exprs
    return ctx


def _build_having_ctx(node: Having, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    ctx["having"].append(_build_condition(node.condition, ctx))
    return ctx


def _build_project_ctx(node: Project, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    cols = []
    for field in node.fields:
        if field in ctx["agg_map"]:
            cols.append(ctx["agg_map"][field])
        else:
            col = _resolve_col(field, ctx)
            label_name = field.replace(".", "_")
            cols.append(col.label(label_name))
    ctx["select_cols"] = cols
    return ctx


def _assemble(ctx: dict):
    expanded = []
    for item in ctx["select_cols"]:
        if hasattr(item, "c"):
            expanded.extend(item.c)
        else:
            expanded.append(item)

    stmt = select(*expanded)
    for from_clause in ctx["froms"]:
        stmt = stmt.select_from(from_clause)
    if ctx["where"]:
        stmt = stmt.where(and_(*ctx["where"]))
    if ctx["groupby"]:
        stmt = stmt.group_by(*ctx["groupby"])
    if ctx["having"]:
        stmt = stmt.having(and_(*ctx["having"]))
    return stmt


def _build_condition(cond: Condition, ctx: dict):
    if isinstance(cond, Compare):
        left_col = _resolve_col(cond.field, ctx)
        right = _resolve_col(cond.value, ctx) if isinstance(cond.value, str) and "." in cond.value else cond.value

        if right is None:
            if cond.op == CmpOp.EQ:
                return left_col.is_(None)
            if cond.op == CmpOp.NEQ:
                return left_col.is_not(None)

        if cond.op == CmpOp.EQ:
            return left_col == right
        if cond.op == CmpOp.NEQ:
            return left_col != right
        if cond.op == CmpOp.GT:
            return left_col > right
        if cond.op == CmpOp.GTE:
            return left_col >= right
        if cond.op == CmpOp.LT:
            return left_col < right
        if cond.op == CmpOp.LTE:
            return left_col <= right
        raise NotImplementedError(f"unknown compare op: {cond.op}")

    if isinstance(cond, And):
        return and_(_build_condition(cond.left, ctx), _build_condition(cond.right, ctx))
    if isinstance(cond, Or):
        return or_(_build_condition(cond.left, ctx), _build_condition(cond.right, ctx))
    if isinstance(cond, Not):
        return not_(_build_condition(cond.child, ctx))
    raise NotImplementedError(f"unknown condition type: {type(cond)}")


def _build_aggregate(agg: Aggregate, ctx: dict):
    if agg.func == AggFunc.COUNT:
        return func.count() if agg.field == "*" else func.count(_resolve_col(agg.field, ctx))

    col = _resolve_col(agg.field, ctx)
    fn_map = {
        AggFunc.SUM: func.sum,
        AggFunc.AVG: func.avg,
        AggFunc.MAX: func.max,
        AggFunc.MIN: func.min,
    }
    return fn_map[agg.func](col)


def _resolve_col(field_name: str, ctx: dict):
    if field_name in ctx.get("agg_map", {}):
        return ctx["agg_map"][field_name]

    if "." in field_name:
        table_alias, col_name = field_name.split(".", 1)
        table = ctx["tables"].get(table_alias)
        if table is None:
            raise KeyError(
                f"[sqlalchemy_orm] unknown alias '{table_alias}', "
                f"known aliases: {list(ctx['tables'].keys())}"
            )
        return table.c[col_name]

    matches = []
    for table in ctx["tables"].values():
        if field_name in table.c:
            matches.append(table.c[field_name])

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(
            f"[sqlalchemy_orm] warning: ambiguous field '{field_name}', "
            "use the first matched column"
        )
        return matches[0]
    raise KeyError(
        f"[sqlalchemy_orm] field '{field_name}' not found, "
        f"known aliases: {list(ctx['tables'].keys())}"
    )
