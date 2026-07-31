"""
translators/sqlalchemy_orm.py

Path 3: IR -> SQLAlchemy Core.
"""

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import MetaData, and_, case, func, literal, not_, or_, select
from sqlalchemy.engine import Engine

from db.connector import get_engine
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
    OrderKey,
    Project,
    QueryNode,
    Scan,
    SelectItem,
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
    if isinstance(node, Distinct):
        return _build_distinct_ctx(node, engine)
    if isinstance(node, OrderBy):
        return _build_orderby_ctx(node, engine)
    if isinstance(node, LimitOffset):
        return _build_limit_offset_ctx(node, engine)
    raise NotImplementedError(f"unknown node type: {type(node)}")


def _base_ctx() -> dict:
    return {
        "froms": [],
        "tables": {},
        "where": [],
        "groupby": [],
        "having": [],
        "agg_map": {},
        "select_cols": [],
        "distinct": False,
        "orderby": [],
        "limit": None,
        "offset": None,
        "engine": None,
    }


def _build_scan_ctx(node: Scan, engine: Engine) -> dict:
    table_alias = _get_table(node.table, engine).alias(node.alias)
    ctx = _base_ctx()
    ctx["engine"] = engine
    ctx["froms"] = [table_alias]
    ctx["tables"] = {node.alias: table_alias}
    ctx["select_cols"] = [table_alias]
    return ctx


def _build_filter_ctx(node: Filter, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    ctx["where"].append(_build_condition(node.condition, ctx))
    return ctx


def _build_join_ctx(node: Join, engine: Engine) -> dict:
    left_ctx = _collect_ctx(node.left, engine)
    right_ctx = _collect_ctx(node.right, engine)

    ctx = _base_ctx()
    ctx["engine"] = engine
    ctx["tables"] = {**left_ctx["tables"], **right_ctx["tables"]}
    ctx["where"] = left_ctx["where"] + right_ctx["where"]
    ctx["groupby"] = left_ctx["groupby"] + right_ctx["groupby"]
    ctx["having"] = left_ctx["having"] + right_ctx["having"]
    ctx["agg_map"] = {**left_ctx["agg_map"], **right_ctx["agg_map"]}
    ctx["select_cols"] = list(left_ctx["select_cols"]) + list(right_ctx["select_cols"])
    ctx["distinct"] = left_ctx["distinct"] or right_ctx["distinct"]
    ctx["orderby"] = left_ctx["orderby"] + right_ctx["orderby"]
    ctx["limit"] = left_ctx["limit"] if left_ctx["limit"] is not None else right_ctx["limit"]
    ctx["offset"] = left_ctx["offset"] if left_ctx["offset"] is not None else right_ctx["offset"]

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
    groupby_cols = [_resolve_expr(field, ctx, prefer_field=True) for field in node.fields]
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
        if isinstance(field, SelectItem):
            cols.append(_resolve_expr(field.expr, ctx, prefer_field=True).label(field.alias))
        elif field in ctx["agg_map"]:
            cols.append(ctx["agg_map"][field])
        else:
            cols.append(_resolve_expr(field, ctx, prefer_field=True).label(str(field).replace(".", "_")))
    ctx["select_cols"] = cols
    return ctx


def _build_distinct_ctx(node: Distinct, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    ctx["distinct"] = True
    return ctx


def _build_orderby_ctx(node: OrderBy, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    ctx["orderby"] = [_build_order_key(key, ctx) for key in node.keys]
    return ctx


def _build_limit_offset_ctx(node: LimitOffset, engine: Engine) -> dict:
    ctx = _collect_ctx(node.child, engine)
    ctx["limit"] = max(0, node.limit)
    ctx["offset"] = max(0, node.offset)
    return ctx


def _assemble(ctx: dict, select_override=None):
    expanded = list(select_override) if select_override is not None else _expand_select_cols(ctx)
    stmt = select(*expanded)
    return _apply_stmt_clauses(stmt, ctx)


def _expand_select_cols(ctx: dict):
    expanded = []
    for item in ctx["select_cols"]:
        if hasattr(item, "c"):
            expanded.extend(item.c)
        else:
            expanded.append(item)
    if not expanded:
        expanded.append(literal(1))
    return expanded


def _apply_stmt_clauses(stmt, ctx: dict):
    if ctx["distinct"]:
        stmt = stmt.distinct()
    for from_clause in ctx["froms"]:
        stmt = stmt.select_from(from_clause)
    if ctx["where"]:
        stmt = stmt.where(and_(*ctx["where"]))
    if ctx["groupby"]:
        stmt = stmt.group_by(*ctx["groupby"])
    if ctx["having"]:
        stmt = stmt.having(and_(*ctx["having"]))
    if ctx["orderby"]:
        stmt = stmt.order_by(*ctx["orderby"])
    if ctx["limit"] is not None:
        stmt = stmt.limit(ctx["limit"])
    if ctx["offset"]:
        stmt = stmt.offset(ctx["offset"])
    return stmt


def _build_condition(cond: Condition, ctx: dict):
    if isinstance(cond, Compare):
        left_col = _resolve_expr(cond.field, ctx, prefer_field=True)
        right = _resolve_expr(cond.value, ctx, prefer_field=True)

        if cond.value is None:
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

    if isinstance(cond, InList):
        left_col = _resolve_expr(cond.field, ctx, prefer_field=True)
        values = [_resolve_expr(value, ctx, prefer_field=True) for value in cond.values]
        return left_col.not_in(values) if cond.negated else left_col.in_(values)

    if isinstance(cond, Between):
        left_col = _resolve_expr(cond.field, ctx, prefer_field=True)
        lower = _resolve_expr(cond.lower, ctx, prefer_field=True)
        upper = _resolve_expr(cond.upper, ctx, prefer_field=True)
        expr = left_col.between(lower, upper)
        return not_(expr) if cond.negated else expr

    if isinstance(cond, Like):
        left_col = _resolve_expr(cond.field, ctx, prefer_field=True)
        expr = left_col.like(cond.pattern)
        return not_(expr) if cond.negated else expr

    if isinstance(cond, Exists):
        sub_ctx = _collect_ctx(cond.subquery, ctx["engine"])
        expr = _assemble(sub_ctx, select_override=[literal(1)]).exists()
        return not_(expr) if cond.negated else expr

    if isinstance(cond, InSubquery):
        left_col = _resolve_expr(cond.field, ctx, prefer_field=True)
        sub_ctx = _collect_ctx(cond.subquery, ctx["engine"])
        sub_stmt = _assemble(sub_ctx)
        if _query_has_limit_offset(cond.subquery):
            subq = sub_stmt.subquery("retorm_in_subq")
            sub_cols = list(subq.c)
            if len(sub_cols) != 1:
                raise ValueError(
                    "[sqlalchemy_orm] IN subquery with LIMIT/OFFSET must return exactly one column"
                )
            sub_stmt = select(sub_cols[0])
        else:
            sub_cols = _expand_select_cols(sub_ctx)
            sub_stmt = select(sub_cols[0])
        expr = left_col.in_(sub_stmt)
        return not_(expr) if cond.negated else expr

    if isinstance(cond, And):
        return and_(_build_condition(cond.left, ctx), _build_condition(cond.right, ctx))
    if isinstance(cond, Or):
        return or_(_build_condition(cond.left, ctx), _build_condition(cond.right, ctx))
    if isinstance(cond, Not):
        return not_(_build_condition(cond.child, ctx))
    raise NotImplementedError(f"unknown condition type: {type(cond)}")


def _build_aggregate(agg: Aggregate, ctx: dict):
    if agg.func == AggFunc.COUNT:
        return func.count() if agg.field == "*" else func.count(_resolve_expr(agg.field, ctx, prefer_field=True))

    col = _resolve_expr(agg.field, ctx, prefer_field=True)
    fn_map = {
        AggFunc.SUM: func.sum,
        AggFunc.AVG: func.avg,
        AggFunc.MAX: func.max,
        AggFunc.MIN: func.min,
    }
    return fn_map[agg.func](col)


def _build_order_key(key: OrderKey, ctx: dict):
    expr = _resolve_expr(key.field, ctx, prefer_field=True)
    return expr.desc() if key.descending else expr.asc()


def _resolve_expr(expr, ctx: dict, prefer_field: bool = False):
    if isinstance(expr, ArithExpr):
        left = _resolve_expr(expr.left, ctx, prefer_field=True)
        right = _resolve_expr(expr.right, ctx, prefer_field=True)
        if expr.op == ArithOp.ADD:
            return left + right
        if expr.op == ArithOp.SUB:
            return left - right
        if expr.op == ArithOp.MUL:
            return left * right
        if expr.op == ArithOp.DIV:
            return left / right
        if expr.op == ArithOp.MOD:
            return left % right
        raise NotImplementedError(f"unknown arithmetic op: {expr.op}")

    if isinstance(expr, CaseWhen):
        whens = []
        for case_item in expr.cases:
            whens.append(
                (
                    _build_condition(case_item.condition, ctx),
                    _resolve_expr(case_item.value, ctx, prefer_field=True),
                )
            )
        return case(*whens, else_=_resolve_expr(expr.else_value, ctx, prefer_field=False))

    if isinstance(expr, str):
        if expr in ctx.get("agg_map", {}):
            return ctx["agg_map"][expr]
        if prefer_field:
            resolved = _maybe_resolve_col(expr, ctx)
            if resolved is not None:
                return resolved
        return literal(expr)

    return literal(expr)


def _maybe_resolve_col(field_name: str, ctx: dict):
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
    return None


def _query_has_limit_offset(node: QueryNode) -> bool:
    if isinstance(node, LimitOffset):
        return True
    if isinstance(node, (Filter, GroupBy, Having, Project, Distinct, OrderBy)):
        return _query_has_limit_offset(node.child)
    if isinstance(node, Join):
        return _query_has_limit_offset(node.left) or _query_has_limit_offset(node.right)
    return False
