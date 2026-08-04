"""Path 3: true ORM execution with mapped classes and Session."""

import os
import random
import sys
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    and_,
    case,
    except_,
    func,
    inspect as sa_inspect,
    intersect,
    join,
    literal,
    not_,
    or_,
    outerjoin,
    select,
    union,
    union_all,
)
from sqlalchemy.orm import Load, aliased, declarative_base, joinedload, relationship, selectinload

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from generator.schema_gen import ColType, Schema
from ir.nodes import (
    AggFunc,
    And,
    ArithExpr,
    ArithOp,
    Between,
    CaseWhen,
    CmpOp,
    Compare,
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
    OrderKey,
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


class UnsupportedTrueORM(Exception):
    """Raised when the current true-ORM implementation does not support an IR shape."""


@dataclass
class TrueORMFacts:
    entity_tables: Dict[str, str] = dc_field(default_factory=dict)
    entity_pk_columns: Dict[str, Tuple[str, ...]] = dc_field(default_factory=dict)
    entity_pks: Dict[str, List[Tuple[Any, ...]]] = dc_field(default_factory=dict)
    duplicate_entity_pks: Dict[str, List[Tuple[Any, ...]]] = dc_field(default_factory=dict)
    loaded_relationships: Dict[str, List[str]] = dc_field(default_factory=dict)
    expected_loaded_relationships: Dict[str, List[str]] = dc_field(default_factory=dict)
    identity_map_size: int = 0
    materialized_entity_count: int = 0


@dataclass
class TrueORMResult:
    rows: Rows
    facts: TrueORMFacts
    compiled_sql: str = ""

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass
class _QueryCtx:
    from_obj: Any
    aliases: Dict[str, Any]
    where_clauses: List[Any] = dc_field(default_factory=list)
    group_by_clauses: List[Any] = dc_field(default_factory=list)
    having_clauses: List[Any] = dc_field(default_factory=list)
    aggregate_exprs: Dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class _SelectSpec:
    project: Project
    distinct: bool = False
    order_keys: List[OrderKey] = dc_field(default_factory=list)
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class _SelectionMeta:
    kind: str
    output_name: str
    alias_name: Optional[str] = None
    entity_cls: Any = None
    entity_ref: Any = None


_MODEL_CACHE: Dict[Tuple, Dict[str, Any]] = {}
_CURRENT_MODELS: Dict[str, Any] = {}
_COVERAGE_KEYS = (
    "relationship_join_used",
    "relationship_join_fallback",
    "explicit_join_used",
    "entity_projection_used",
    "entity_materialization_used",
    "entity_scalar_mix_used",
    "joinedload_used",
    "selectinload_used",
    "relationship_touch_used",
    "self_alias_used",
    "set_query_used",
    "scalar_subquery_used",
    "window_expr_used",
    "derived_table_used",
    "limit_in_subquery_wrap_used",
    "fault_injection_triggered",
)
_TRUE_ORM_COVERAGE: Dict[str, int] = {key: 0 for key in _COVERAGE_KEYS}


def reset_true_orm_coverage() -> None:
    for key in _COVERAGE_KEYS:
        _TRUE_ORM_COVERAGE[key] = 0


def get_true_orm_coverage_snapshot() -> Dict[str, int]:
    return dict(_TRUE_ORM_COVERAGE)


def _bump_coverage(key: str, amount: int = 1) -> None:
    if not getattr(config, "TRUE_ORM_COVERAGE_ENABLED", True):
        return
    if key in _TRUE_ORM_COVERAGE:
        _TRUE_ORM_COVERAGE[key] += amount


def execute(ir: QueryNode, schema: Schema) -> TrueORMResult:
    from db.connector import get_session

    ok, reason = supports_true_orm(ir)
    if not ok:
        raise UnsupportedTrueORM(reason)

    models = _get_models(schema)
    global _CURRENT_MODELS
    _CURRENT_MODELS = models
    stmt, selection_meta = _build_query(ir, models, outer_aliases=None)

    with get_session() as session:
        stmt, needs_unique, expected_loaded_relationships = _apply_loader_options(stmt, selection_meta)
        compiled_sql = _maybe_sample_compiled_sql(stmt, session)
        result = session.execute(stmt)
        if needs_unique:
            result = result.unique()
        result_rows = result.all()
        facts = _collect_true_orm_facts(
            result_rows,
            selection_meta,
            session,
            expected_loaded_relationships,
        )
        flattened = [_flatten_result_row(row, selection_meta) for row in result_rows]
        if config.TRUE_ORM_TOUCH_RELATIONSHIPS:
            _touch_relationships(result_rows, selection_meta)
        return TrueORMResult(rows=flattened, facts=facts, compiled_sql=compiled_sql)


def supports_true_orm(ir: QueryNode) -> Tuple[bool, str]:
    try:
        _validate_query(ir)
        return True, ""
    except UnsupportedTrueORM as exc:
        return False, str(exc)


def reset_model_cache() -> None:
    _MODEL_CACHE.clear()


def _build_query(
    root: QueryNode,
    models: Dict[str, Any],
    outer_aliases: Optional[Dict[str, Any]],
    derived_labels: bool = False,
    allow_entity_projection: bool = True,
):
    if isinstance(root, SetQuery):
        _bump_coverage("set_query_used")
        left_stmt, left_meta = _build_query(
            root.left,
            models,
            outer_aliases=outer_aliases,
            derived_labels=derived_labels,
            allow_entity_projection=False,
        )
        right_stmt, right_meta = _build_query(
            root.right,
            models,
            outer_aliases=outer_aliases,
            derived_labels=derived_labels,
            allow_entity_projection=False,
        )
        if len(left_meta) != len(right_meta):
            raise UnsupportedTrueORM("set-op branches must have the same projection width")
        if root.op == SetOp.UNION:
            stmt = union_all(left_stmt, right_stmt) if root.all else union(left_stmt, right_stmt)
        elif root.op == SetOp.INTERSECT:
            if root.all:
                raise UnsupportedTrueORM("INTERSECT ALL is not implemented in true ORM yet")
            stmt = intersect(left_stmt, right_stmt)
        elif root.op == SetOp.EXCEPT:
            if root.all:
                raise UnsupportedTrueORM("EXCEPT ALL is not implemented in true ORM yet")
            stmt = except_(left_stmt, right_stmt)
        else:
            raise UnsupportedTrueORM(f"unsupported set op: {root.op}")
        return stmt, left_meta

    spec = _extract_select_spec(root)
    ctx = _build_rel_ctx(spec.project.child, models, outer_aliases=outer_aliases)
    select_items, selection_meta, projection_aliases = _build_projection_items(
        spec.project.fields,
        ctx,
        derived_labels=derived_labels,
        allow_entity_projection=allow_entity_projection,
    )

    stmt = select(*select_items).select_from(ctx.from_obj)
    if ctx.where_clauses:
        stmt = stmt.where(and_(*ctx.where_clauses))
    if ctx.group_by_clauses:
        stmt = stmt.group_by(*ctx.group_by_clauses)
    if ctx.having_clauses:
        stmt = stmt.having(and_(*ctx.having_clauses))
    if spec.distinct:
        stmt = stmt.distinct()
    if spec.order_keys:
        stmt = stmt.order_by(
            *[
                _build_order_key(key, ctx.aliases, ctx.aggregate_exprs, projection_aliases)
                for key in spec.order_keys
            ]
        )
    if spec.limit is not None:
        stmt = stmt.limit(max(0, spec.limit))
        applied_offset = spec.offset
        if _fault_mode() == "drop_offset":
            applied_offset = 0
            if spec.offset:
                _bump_coverage("fault_injection_triggered")
        if applied_offset:
            stmt = stmt.offset(max(0, applied_offset))
    return stmt, selection_meta


def _extract_select_spec(node: QueryNode) -> _SelectSpec:
    distinct = False
    order_keys: List[OrderKey] = []
    limit: Optional[int] = None
    offset = 0
    cur = node

    while True:
        if isinstance(cur, Distinct):
            distinct = True
            cur = cur.child
            continue
        if isinstance(cur, OrderBy):
            order_keys = cur.keys + order_keys
            cur = cur.child
            continue
        if isinstance(cur, LimitOffset):
            if limit is not None:
                raise UnsupportedTrueORM("multiple LIMIT/OFFSET wrappers are not supported yet")
            limit = cur.limit
            offset = cur.offset
            cur = cur.child
            continue
        if isinstance(cur, Project):
            return _SelectSpec(
                project=cur,
                distinct=distinct,
                order_keys=order_keys,
                limit=limit,
                offset=offset,
            )
        raise UnsupportedTrueORM(
            "true ORM expects Project as the base query after optional "
            f"Distinct/OrderBy/LimitOffset wrappers, got {type(cur).__name__}"
        )


def _validate_query(node: QueryNode) -> None:
    if isinstance(node, SetQuery):
        _validate_query(node.left)
        _validate_query(node.right)
        return
    spec = _extract_select_spec(node)
    _validate_project(spec.project)


def _validate_project(node: Project) -> None:
    for field in node.fields:
        if isinstance(field, SelectItem):
            _validate_value_expr(field.expr)
            continue
        if isinstance(field, str):
            continue
        raise UnsupportedTrueORM(f"unsupported projection field: {field!r}")
    _validate_rel_node(node.child)


def _validate_rel_node(node: QueryNode) -> None:
    if isinstance(node, (Scan, DerivedTable)):
        if isinstance(node, DerivedTable):
            _validate_query(node.subquery)
        return
    if isinstance(node, Filter):
        _validate_condition(node.condition)
        _validate_rel_node(node.child)
        return
    if isinstance(node, Join):
        _validate_condition(node.on)
        _validate_rel_node(node.left)
        _validate_rel_node(node.right)
        return
    if isinstance(node, GroupBy):
        for field in node.fields:
            _validate_value_expr(field)
        for agg in node.aggregates:
            if agg.field != "*":
                _validate_value_expr(agg.field)
        _validate_rel_node(node.child)
        return
    if isinstance(node, Having):
        _validate_condition(node.condition)
        _validate_rel_node(node.child)
        return
    raise UnsupportedTrueORM(f"unsupported relational node for true ORM: {type(node).__name__}")


def _validate_condition(cond) -> None:
    if isinstance(cond, Compare):
        _validate_value_expr(cond.field)
        _validate_value_expr(cond.value)
        return
    if isinstance(cond, InList):
        _validate_value_expr(cond.field)
        for value in cond.values:
            _validate_value_expr(value)
        return
    if isinstance(cond, Between):
        _validate_value_expr(cond.field)
        _validate_value_expr(cond.lower)
        _validate_value_expr(cond.upper)
        return
    if isinstance(cond, Like):
        _validate_value_expr(cond.field)
        return
    if isinstance(cond, Exists):
        _validate_query(cond.subquery)
        return
    if isinstance(cond, InSubquery):
        _validate_value_expr(cond.field)
        _validate_query(cond.subquery)
        return
    if isinstance(cond, (And, Or)):
        _validate_condition(cond.left)
        _validate_condition(cond.right)
        return
    if isinstance(cond, Not):
        _validate_condition(cond.child)
        return
    raise UnsupportedTrueORM(f"unsupported condition for true ORM: {type(cond).__name__}")


def _validate_value_expr(expr) -> None:
    if expr is None or isinstance(expr, (int, float, str)):
        return
    if isinstance(expr, ArithExpr):
        _validate_value_expr(expr.left)
        _validate_value_expr(expr.right)
        return
    if isinstance(expr, CaseWhen):
        for case_item in expr.cases:
            _validate_condition(case_item.condition)
            _validate_value_expr(case_item.value)
        _validate_value_expr(expr.else_value)
        return
    if isinstance(expr, ScalarSubquery):
        _validate_query(expr.subquery)
        return
    if isinstance(expr, WindowExpr):
        if expr.field not in (None, "*"):
            _validate_value_expr(expr.field)
        for item in expr.partition_by:
            _validate_value_expr(item)
        for key in expr.order_by:
            _validate_value_expr(key.field)
        return
    raise UnsupportedTrueORM(f"unsupported value expression for true ORM: {type(expr).__name__}")


def _build_rel_ctx(
    node: QueryNode,
    models: Dict[str, Any],
    outer_aliases: Optional[Dict[str, Any]],
) -> _QueryCtx:
    outer_aliases = outer_aliases or {}

    if isinstance(node, Scan):
        model_alias = aliased(models[node.table], name=node.alias)
        if any(_entity_mapper_class(obj).__table__.name == node.table for obj in outer_aliases.values() if not hasattr(obj, "c")):
            _bump_coverage("self_alias_used")
        aliases = dict(outer_aliases)
        aliases[node.alias] = model_alias
        return _QueryCtx(from_obj=model_alias, aliases=aliases)

    if isinstance(node, DerivedTable):
        _bump_coverage("derived_table_used")
        sub_stmt, sub_meta = _build_query(
            node.subquery,
            models,
            outer_aliases=outer_aliases,
            derived_labels=True,
        )
        subquery_obj = sub_stmt.subquery(node.alias)
        aliases = dict(outer_aliases)
        aliases[node.alias] = subquery_obj
        return _QueryCtx(from_obj=subquery_obj, aliases=aliases)

    if isinstance(node, Filter):
        child = _build_rel_ctx(node.child, models, outer_aliases=outer_aliases)
        child.where_clauses.append(
            _build_condition(node.condition, child.aliases, child.aggregate_exprs)
        )
        return child

    if isinstance(node, Join):
        left = _build_rel_ctx(node.left, models, outer_aliases=outer_aliases)
        right = _build_rel_ctx(node.right, models, outer_aliases=left.aliases)
        aliases = dict(left.aliases)
        aliases.update(right.aliases)
        aggregate_exprs = {**left.aggregate_exprs, **right.aggregate_exprs}
        join_obj = _build_join_obj(node, left, right, aliases, aggregate_exprs)
        return _QueryCtx(
            from_obj=join_obj,
            aliases=aliases,
            where_clauses=left.where_clauses + right.where_clauses,
            group_by_clauses=left.group_by_clauses + right.group_by_clauses,
            having_clauses=left.having_clauses + right.having_clauses,
            aggregate_exprs=aggregate_exprs,
        )

    if isinstance(node, GroupBy):
        child = _build_rel_ctx(node.child, models, outer_aliases=outer_aliases)
        aggregate_exprs = dict(child.aggregate_exprs)
        for agg in node.aggregates:
            aggregate_exprs[agg.alias] = _build_aggregate_expr(agg, child.aliases, aggregate_exprs)
        return _QueryCtx(
            from_obj=child.from_obj,
            aliases=child.aliases,
            where_clauses=list(child.where_clauses),
            group_by_clauses=[
                _resolve_expr(
                    field,
                    child.aliases,
                    aggregate_exprs,
                    resolve_unqualified_field=True,
                )
                for field in node.fields
            ],
            having_clauses=list(child.having_clauses),
            aggregate_exprs=aggregate_exprs,
        )

    if isinstance(node, Having):
        child = _build_rel_ctx(node.child, models, outer_aliases=outer_aliases)
        child.having_clauses.append(
            _build_condition(node.condition, child.aliases, child.aggregate_exprs)
        )
        return child

    raise UnsupportedTrueORM(f"unsupported relational node: {type(node).__name__}")


def _build_join_obj(node: Join, left: _QueryCtx, right: _QueryCtx, aliases, aggregate_exprs):
    relationship_join = _try_relationship_join(node, left, right)
    if relationship_join is not None:
        return relationship_join

    if config.TRUE_ORM_JOIN_MODE in ("relationship", "relationship_preferred"):
        _bump_coverage("relationship_join_fallback")

    on_clause = _build_condition(node.on, aliases, aggregate_exprs)
    if node.join_type == JoinType.LEFT and _fault_mode() == "inner_for_left_join":
        _bump_coverage("fault_injection_triggered")
        _bump_coverage("explicit_join_used")
        return join(left.from_obj, right.from_obj, on_clause)
    if node.join_type == JoinType.LEFT:
        _bump_coverage("explicit_join_used")
        return outerjoin(left.from_obj, right.from_obj, on_clause)
    _bump_coverage("explicit_join_used")
    return join(left.from_obj, right.from_obj, on_clause)


def _try_relationship_join(node: Join, left: _QueryCtx, right: _QueryCtx):
    if config.TRUE_ORM_JOIN_MODE not in ("relationship", "relationship_preferred"):
        return None
    if not isinstance(node.on, Compare) or node.on.op != CmpOp.EQ:
        return None
    if not isinstance(node.on.field, str) or not isinstance(node.on.value, str):
        return None

    for source_name, target_name, field_name, value_name in (
        (node.on.field, node.on.value, node.on.field, node.on.value),
        (node.on.value, node.on.field, node.on.value, node.on.field),
    ):
        source_alias_name = source_name.split(".", 1)[0] if "." in source_name else None
        target_alias_name = target_name.split(".", 1)[0] if "." in target_name else None
        source_alias = left.aliases.get(source_alias_name) or right.aliases.get(source_alias_name)
        target_alias = left.aliases.get(target_alias_name) or right.aliases.get(target_alias_name)
        if source_alias is None or target_alias is None:
            continue
        rel_name = _find_relationship_name(source_alias, field_name, value_name)
        if not rel_name:
            continue
        rel_attr = getattr(source_alias, rel_name)
        rel_target = rel_attr.of_type(target_alias)
        on_clause = rel_target.__clause_element__()
        _bump_coverage("relationship_join_used")
        if node.join_type == JoinType.LEFT:
            return outerjoin(left.from_obj, right.from_obj, on_clause)
        return join(left.from_obj, right.from_obj, on_clause)

    if config.TRUE_ORM_JOIN_MODE == "relationship":
        raise UnsupportedTrueORM("relationship join requested, but join condition did not map to a generated relationship")
    return None


def _find_relationship_name(source_alias, source_field_name: str, target_field_name: str):
    mapper_cls = getattr(source_alias, "_aliased_insp", None)
    if mapper_cls is not None:
        base_cls = mapper_cls.mapper.class_
    else:
        base_cls = source_alias
    rel_map = getattr(base_cls, "__retorm_relationships__", {})
    source_col = source_field_name.split(".", 1)[1] if "." in source_field_name else source_field_name
    target_col = target_field_name.split(".", 1)[1] if "." in target_field_name else target_field_name
    for rel_name, meta in rel_map.items():
        if meta["source_col"] == source_col and meta["target_col"] == target_col:
            return rel_name
    return None


def _build_projection_items(fields, ctx: _QueryCtx, derived_labels: bool = False, allow_entity_projection: bool = True):
    items = []
    meta: List[_SelectionMeta] = []
    projection_aliases: Dict[str, Any] = {}
    seen_labels: Dict[str, int] = {}
    entity_projection_aliases = (
        _find_full_entity_projection_aliases(fields, ctx.aliases)
        if allow_entity_projection and not derived_labels and config.TRUE_ORM_ENTITY_PROJECTION
        else {}
    )
    emitted_entity_aliases = set()

    for field in fields:
        if isinstance(field, SelectItem):
            expr = _resolve_expr(field.expr, ctx.aliases, ctx.aggregate_exprs, projection_aliases)
            label = _next_projection_label(field.alias, seen_labels)
            labeled = expr.label(label)
            items.append(labeled)
            meta.append(_SelectionMeta(kind="scalar", output_name=label))
            projection_aliases[field.alias] = labeled
            continue

        entity_alias = _entity_projection_alias_for_field(field, entity_projection_aliases)
        if entity_alias is not None:
            if entity_alias in emitted_entity_aliases:
                continue
            emitted_entity_aliases.add(entity_alias)
            entity = ctx.aliases[entity_alias]
            items.append(entity)
            _bump_coverage("entity_projection_used")
            _bump_coverage("entity_materialization_used")
            meta.append(
                _SelectionMeta(
                    kind="entity",
                    output_name=entity_alias,
                    alias_name=entity_alias,
                    entity_cls=_entity_mapper_class(entity),
                    entity_ref=entity,
                )
            )
            continue

        if (
            allow_entity_projection
            and isinstance(field, str)
            and field in ctx.aliases
            and config.TRUE_ORM_ENTITY_PROJECTION
        ):
            if derived_labels:
                raise UnsupportedTrueORM("derived-table subqueries cannot project whole ORM entities")
            entity = ctx.aliases[field]
            items.append(entity)
            _bump_coverage("entity_projection_used")
            _bump_coverage("entity_materialization_used")
            meta.append(
                _SelectionMeta(
                    kind="entity",
                    output_name=field,
                    alias_name=field,
                    entity_cls=_entity_mapper_class(entity),
                    entity_ref=entity,
                )
            )
            continue

        expr = _resolve_expr(
            field,
            ctx.aliases,
            ctx.aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        if derived_labels or _is_derived_table_field(field, ctx.aliases):
            base_label = _derived_field_label(field)
        else:
            base_label = _default_field_label(field)
        label = _next_projection_label(base_label, seen_labels)
        labeled = expr.label(label)
        items.append(labeled)
        output_name = field if isinstance(field, str) and _is_derived_table_field(field, ctx.aliases) else label
        meta.append(_SelectionMeta(kind="scalar", output_name=output_name))
        if isinstance(field, str):
            projection_aliases[field] = labeled

    if any(item.kind == "entity" for item in meta) and any(item.kind == "scalar" for item in meta):
        _bump_coverage("entity_scalar_mix_used")

    return items, meta, projection_aliases


def _build_order_key(key: OrderKey, aliases, aggregate_exprs, projection_aliases):
    expr = _resolve_expr(
        key.field,
        aliases,
        aggregate_exprs,
        projection_aliases,
        resolve_unqualified_field=True,
    )
    descending = key.descending
    if _fault_mode() == "reverse_order":
        descending = not descending
        _bump_coverage("fault_injection_triggered")
    return expr.desc() if descending else expr.asc()


def _build_aggregate_expr(agg, aliases, aggregate_exprs):
    if agg.field == "*":
        if agg.func != AggFunc.COUNT:
            raise UnsupportedTrueORM(f"{agg.func.value}(*) is not supported")
        return func.count()

    if agg.func == AggFunc.COUNT and _fault_mode() == "count_star":
        _bump_coverage("fault_injection_triggered")
        return func.count()

    value_expr = _resolve_expr(
        agg.field,
        aliases,
        aggregate_exprs,
        resolve_unqualified_field=True,
    )
    if agg.func == AggFunc.SUM:
        return func.sum(value_expr)
    if agg.func == AggFunc.COUNT:
        return func.count(value_expr)
    if agg.func == AggFunc.AVG:
        return func.avg(value_expr)
    if agg.func == AggFunc.MAX:
        return func.max(value_expr)
    if agg.func == AggFunc.MIN:
        return func.min(value_expr)
    raise UnsupportedTrueORM(f"unsupported aggregate function: {agg.func}")


def _build_condition(cond, aliases, aggregate_exprs, projection_aliases=None):
    if isinstance(cond, Compare):
        left = _resolve_expr(
            cond.field,
            aliases,
            aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        right = _resolve_expr(cond.value, aliases, aggregate_exprs, projection_aliases)
        if cond.value is None:
            if _fault_mode() == "null_eq_false":
                _bump_coverage("fault_injection_triggered")
                if cond.op == CmpOp.EQ:
                    return literal(False)
                if cond.op == CmpOp.NEQ:
                    return literal(True)
            if cond.op == CmpOp.EQ:
                return left.is_(None)
            if cond.op == CmpOp.NEQ:
                return left.is_not(None)
        if cond.op == CmpOp.EQ:
            return left == right
        if cond.op == CmpOp.NEQ:
            return left != right
        if cond.op == CmpOp.GT:
            return left > right
        if cond.op == CmpOp.GTE:
            return left >= right
        if cond.op == CmpOp.LT:
            return left < right
        if cond.op == CmpOp.LTE:
            return left <= right
        raise UnsupportedTrueORM(f"unknown compare op: {cond.op}")

    if isinstance(cond, InList):
        left = _resolve_expr(
            cond.field,
            aliases,
            aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        values = [_resolve_expr(value, aliases, aggregate_exprs, projection_aliases) for value in cond.values]
        expr = left.in_(values)
        return not_(expr) if cond.negated else expr

    if isinstance(cond, Between):
        left = _resolve_expr(
            cond.field,
            aliases,
            aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        lower = _resolve_expr(cond.lower, aliases, aggregate_exprs, projection_aliases)
        upper = _resolve_expr(cond.upper, aliases, aggregate_exprs, projection_aliases)
        expr = left.between(lower, upper)
        return not_(expr) if cond.negated else expr

    if isinstance(cond, Like):
        left = _resolve_expr(
            cond.field,
            aliases,
            aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        expr = left.like(cond.pattern)
        return not_(expr) if cond.negated else expr

    if isinstance(cond, Exists):
        sub_stmt, _ = _build_query(cond.subquery, _CURRENT_MODELS, outer_aliases=aliases)
        expr = sub_stmt.exists()
        return not_(expr) if cond.negated else expr

    if isinstance(cond, InSubquery):
        left = _resolve_expr(
            cond.field,
            aliases,
            aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        sub_stmt, _ = _build_query(cond.subquery, _CURRENT_MODELS, outer_aliases=aliases)
        expr = left.in_(_wrap_in_subquery_if_needed(cond.subquery, sub_stmt))
        return not_(expr) if cond.negated else expr

    if isinstance(cond, And):
        return and_(
            _build_condition(cond.left, aliases, aggregate_exprs, projection_aliases),
            _build_condition(cond.right, aliases, aggregate_exprs, projection_aliases),
        )
    if isinstance(cond, Or):
        return or_(
            _build_condition(cond.left, aliases, aggregate_exprs, projection_aliases),
            _build_condition(cond.right, aliases, aggregate_exprs, projection_aliases),
        )
    if isinstance(cond, Not):
        return not_(_build_condition(cond.child, aliases, aggregate_exprs, projection_aliases))

    raise UnsupportedTrueORM(f"unsupported condition: {type(cond)}")


def _resolve_expr(
    expr,
    aliases,
    aggregate_exprs,
    projection_aliases=None,
    resolve_unqualified_field: bool = False,
):
    projection_aliases = projection_aliases or {}

    if isinstance(expr, str):
        if expr in projection_aliases:
            return projection_aliases[expr]
        if expr in aggregate_exprs:
            return aggregate_exprs[expr]
        if "." in expr:
            return _resolve_field(expr, aliases)
        if expr in aliases and config.TRUE_ORM_ENTITY_PROJECTION:
            raise UnsupportedTrueORM("entity aliases can only be projected directly, not used as scalar expressions")
        if resolve_unqualified_field:
            return _resolve_unqualified_field(expr, aliases)
        return literal(expr)

    if expr is None or isinstance(expr, (int, float)):
        return literal(expr)

    if isinstance(expr, ArithExpr):
        left = _resolve_expr(expr.left, aliases, aggregate_exprs, projection_aliases)
        right = _resolve_expr(expr.right, aliases, aggregate_exprs, projection_aliases)
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
        raise UnsupportedTrueORM(f"unsupported arithmetic op: {expr.op}")

    if isinstance(expr, CaseWhen):
        whens = [
            (
                _build_condition(case_item.condition, aliases, aggregate_exprs, projection_aliases),
                _resolve_expr(case_item.value, aliases, aggregate_exprs, projection_aliases),
            )
            for case_item in expr.cases
        ]
        else_expr = _resolve_expr(expr.else_value, aliases, aggregate_exprs, projection_aliases)
        return case(*whens, else_=else_expr)

    if isinstance(expr, ScalarSubquery):
        _bump_coverage("scalar_subquery_used")
        sub_stmt, _ = _build_query(expr.subquery, _CURRENT_MODELS, outer_aliases=aliases)
        return sub_stmt.scalar_subquery()

    if isinstance(expr, WindowExpr):
        return _build_window_expr(expr, aliases, aggregate_exprs, projection_aliases)

    raise UnsupportedTrueORM(f"unsupported value expression: {type(expr)}")


def _build_window_expr(expr: WindowExpr, aliases, aggregate_exprs, projection_aliases):
    _bump_coverage("window_expr_used")
    if expr.func in {WindowFunc.ROW_NUMBER, WindowFunc.RANK, WindowFunc.DENSE_RANK}:
        base = getattr(func, expr.func.value.lower())()
    else:
        if expr.field == "*":
            if expr.func != WindowFunc.COUNT:
                raise UnsupportedTrueORM(f"{expr.func.value}(*) over() is not supported")
            base = func.count()
        else:
            field_expr = _resolve_expr(
                expr.field,
                aliases,
                aggregate_exprs,
                projection_aliases,
                resolve_unqualified_field=True,
            )
            func_name = expr.func.value.lower()
            base = getattr(func, func_name)(field_expr)
    partition_by = [
        _resolve_expr(
            item,
            aliases,
            aggregate_exprs,
            projection_aliases,
            resolve_unqualified_field=True,
        )
        for item in expr.partition_by
    ]
    order_by = [
        _build_order_key(item, aliases, aggregate_exprs, projection_aliases)
        for item in expr.order_by
    ]
    return base.over(partition_by=partition_by or None, order_by=order_by or None)


def _resolve_field(field_name: str, aliases: Dict[str, Any]):
    table_alias, col_name = field_name.split(".", 1)
    obj = aliases.get(table_alias)
    if obj is None:
        raise UnsupportedTrueORM(f"unknown ORM alias: {table_alias!r}")
    if hasattr(obj, "c"):
        return getattr(obj.c, col_name)
    return getattr(obj, col_name)


def _wrap_in_subquery_if_needed(node: QueryNode, sub_stmt):
    if not _query_has_limit_offset(node):
        return sub_stmt

    _bump_coverage("limit_in_subquery_wrap_used")
    derived = sub_stmt.subquery("retorm_in_subq")
    columns = list(derived.c)
    if len(columns) != 1:
        raise UnsupportedTrueORM(
            f"IN subquery must project exactly one column, got {len(columns)}"
        )
    return select(columns[0]).select_from(derived)


def _query_has_limit_offset(node: QueryNode) -> bool:
    if isinstance(node, LimitOffset):
        return True
    if isinstance(node, DerivedTable):
        return _query_has_limit_offset(node.subquery)
    if isinstance(node, SetQuery):
        return _query_has_limit_offset(node.left) or _query_has_limit_offset(node.right)
    if isinstance(node, (Filter, GroupBy, Having, Project, Distinct, OrderBy)):
        return _query_has_limit_offset(node.child)
    if isinstance(node, Join):
        return _query_has_limit_offset(node.left) or _query_has_limit_offset(node.right)
    return False


def _resolve_unqualified_field(field_name: str, aliases: Dict[str, Any]):
    matches = []
    for alias_name, obj in aliases.items():
        if hasattr(obj, "c"):
            if field_name in obj.c:
                matches.append((alias_name, getattr(obj.c, field_name)))
            continue
        if hasattr(obj, field_name):
            matches.append((alias_name, getattr(obj, field_name)))

    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        alias_list = ", ".join(alias_name for alias_name, _ in matches)
        raise UnsupportedTrueORM(
            f"ambiguous unqualified field {field_name!r}; candidates: {alias_list}"
        )
    raise UnsupportedTrueORM(f"unknown unqualified field {field_name!r}")


def _find_full_entity_projection_aliases(fields, aliases: Dict[str, Any]) -> Dict[str, set]:
    projected_cols: Dict[str, set] = {}
    for field in fields:
        if not isinstance(field, str) or "." not in field:
            continue
        alias_name, col_name = field.split(".", 1)
        obj = aliases.get(alias_name)
        if obj is None or hasattr(obj, "c"):
            continue
        projected_cols.setdefault(alias_name, set()).add(col_name)

    result: Dict[str, set] = {}
    for alias_name, cols in projected_cols.items():
        entity = aliases.get(alias_name)
        if entity is None:
            continue
        entity_cls = _entity_mapper_class(entity)
        entity_cols = {col.name for col in entity_cls.__table__.columns}
        if entity_cols and entity_cols.issubset(cols):
            result[alias_name] = entity_cols
    return result


def _entity_projection_alias_for_field(field, entity_projection_aliases: Dict[str, set]) -> Optional[str]:
    if not isinstance(field, str) or "." not in field:
        return None
    alias_name, col_name = field.split(".", 1)
    cols = entity_projection_aliases.get(alias_name)
    if cols is None or col_name not in cols:
        return None
    return alias_name


def _is_derived_table_field(field_name: Any, aliases: Dict[str, Any]) -> bool:
    if not isinstance(field_name, str) or "." not in field_name:
        return False
    table_alias, _ = field_name.split(".", 1)
    obj = aliases.get(table_alias)
    if obj is None or not hasattr(obj, "c"):
        return False
    return not hasattr(obj, "_aliased_insp")


def _flatten_result_row(row, selection_meta: List[_SelectionMeta]) -> Row:
    out = {}
    for idx, meta in enumerate(selection_meta):
        value = row[idx]
        if meta.kind == "scalar":
            out[meta.output_name] = value
            continue
        if meta.kind == "entity":
            if value is None:
                for col in meta.entity_cls.__table__.columns:
                    out[f"{meta.alias_name}.{col.name}"] = None
            else:
                for col in meta.entity_cls.__table__.columns:
                    out[f"{meta.alias_name}.{col.name}"] = getattr(value, col.name)
            continue
        raise UnsupportedTrueORM(f"unknown selection meta kind: {meta.kind}")
    return out


def _apply_loader_options(stmt, selection_meta):
    strategy = getattr(config, "TRUE_ORM_LOADER_STRATEGY", "off")
    if strategy == "off":
        return stmt, False, {}
    options = []
    needs_unique = False
    expected_loaded_relationships: Dict[str, List[str]] = {}
    for meta in selection_meta:
        if meta.kind != "entity" or meta.entity_ref is None:
            continue
        entity_root = meta.entity_ref
        entity_cls = meta.entity_cls or _entity_mapper_class(entity_root)
        rel_names = list(getattr(entity_cls, "__retorm_relationships__", {}).keys())
        for rel_name in rel_names[:1]:
            rel_attr = getattr(entity_root, rel_name)
            rel_prop = getattr(rel_attr, "property", None)
            expected_loaded_relationships.setdefault(meta.alias_name or meta.output_name, []).append(rel_name)
            if strategy == "joined":
                # `joinedload()` over collection relationships forces ORM-level
                # row de-duplication, which changes the SQL row multiplicity we
                # want to compare against. Keep scalar joins as joined eager
                # loading, but fall back to select-in for collections so the
                # query result shape stays comparable to raw SQL.
                if rel_prop is not None and getattr(rel_prop, "uselist", False):
                    options.append(Load(entity_root).selectinload(rel_attr))
                    _bump_coverage("selectinload_used")
                else:
                    options.append(Load(entity_root).joinedload(rel_attr))
                    _bump_coverage("joinedload_used")
            elif strategy == "selectin":
                options.append(Load(entity_root).selectinload(rel_attr))
                _bump_coverage("selectinload_used")
    if options:
        stmt = stmt.options(*options)
    return stmt, needs_unique, {
        alias_name: sorted(set(rel_names))
        for alias_name, rel_names in expected_loaded_relationships.items()
    }


def _collect_true_orm_facts(
    result_rows,
    selection_meta: List[_SelectionMeta],
    session,
    expected_loaded_relationships: Dict[str, List[str]],
) -> TrueORMFacts:
    facts = TrueORMFacts(
        expected_loaded_relationships={
            alias_name: list(rel_names)
            for alias_name, rel_names in expected_loaded_relationships.items()
        },
        identity_map_size=len(session.identity_map),
    )
    duplicate_lists: Dict[str, List[Tuple[Any, ...]]] = {}

    for meta in selection_meta:
        if meta.kind != "entity":
            continue
        alias_name = meta.alias_name or meta.output_name
        entity_cls = meta.entity_cls
        pk_cols = tuple(col.name for col in entity_cls.__table__.primary_key.columns)
        facts.entity_tables.setdefault(alias_name, entity_cls.__table__.name)
        facts.entity_pk_columns.setdefault(alias_name, pk_cols)

    for row in result_rows:
        for idx, meta in enumerate(selection_meta):
            if meta.kind != "entity":
                continue
            alias_name = meta.alias_name or meta.output_name
            entity_cls = meta.entity_cls
            value = row[idx]
            pk_cols = facts.entity_pk_columns[alias_name]
            if value is None:
                continue

            facts.materialized_entity_count += 1
            pk_tuple = tuple(getattr(value, col_name) for col_name in pk_cols)
            seen = facts.entity_pks.setdefault(alias_name, [])
            if pk_tuple in seen:
                duplicate_lists.setdefault(alias_name, []).append(pk_tuple)
            seen.append(pk_tuple)
            loaded = facts.loaded_relationships.setdefault(alias_name, [])
            loaded.extend(_loaded_relationship_names(value, entity_cls))

    for alias_name, duplicates in duplicate_lists.items():
        facts.duplicate_entity_pks[alias_name] = duplicates.copy()

    for alias_name, rel_names in list(facts.loaded_relationships.items()):
        facts.loaded_relationships[alias_name] = sorted(set(rel_names))

    return facts


def _loaded_relationship_names(entity_obj, entity_cls) -> List[str]:
    state = sa_inspect(entity_obj)
    unloaded = set(getattr(state, "unloaded", set()) or set())
    rel_names = list(getattr(entity_cls, "__retorm_relationships__", {}).keys())
    return [rel_name for rel_name in rel_names if rel_name not in unloaded]


def _maybe_sample_compiled_sql(stmt, session) -> str:
    rate = max(0.0, float(getattr(config, "TRUE_ORM_SQL_SAMPLE_RATE", 0.0) or 0.0))
    if rate <= 0.0 or random.random() > rate:
        return ""
    bind = session.get_bind()
    if bind is not None:
        compiled = stmt.compile(
            dialect=bind.dialect,
            compile_kwargs={"literal_binds": True},
        )
    else:
        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    return str(compiled)


def _fault_mode() -> str:
    mode = getattr(config, "TRUE_ORM_FAULT_INJECTION", "off")
    return mode or "off"


def _touch_relationships(result_rows, selection_meta):
    for row in result_rows:
        for idx, meta in enumerate(selection_meta):
            if meta.kind != "entity":
                continue
            value = row[idx]
            if value is None:
                continue
            for rel_name in getattr(meta.entity_cls, "__retorm_relationships__", {}).keys():
                _bump_coverage("relationship_touch_used")
                getattr(value, rel_name, None)


def _entity_mapper_class(entity):
    mapper = getattr(entity, "_aliased_insp", None)
    if mapper is not None:
        return mapper.mapper.class_
    return entity


def _default_field_label(field_name: str) -> str:
    if isinstance(field_name, str) and "." in field_name:
        alias, col_name = field_name.split(".", 1)
        return f"{alias}_{col_name}"
    if isinstance(field_name, str):
        return field_name
    return repr(field_name)


def _derived_field_label(field_name: str) -> str:
    if isinstance(field_name, str) and "." in field_name:
        return field_name.split(".", 1)[1]
    if isinstance(field_name, str):
        return field_name
    return repr(field_name)


def _next_projection_label(base: str, seen_labels: Dict[str, int]) -> str:
    count = seen_labels.get(base, 0) + 1
    seen_labels[base] = count
    if count == 1:
        return base
    if "_" in base:
        prefix, rest = base.split("_", 1)
        return f"{prefix}{count}_{rest}"
    return f"{base}_{count}"


def _get_models(schema: Schema) -> Dict[str, Any]:
    signature = _schema_signature(schema)
    cached = _MODEL_CACHE.get(signature)
    if cached is not None:
        return cached

    base = declarative_base()
    models: Dict[str, Any] = {}

    for table in schema.tables:
        fk_by_col = {fk.src_col: fk for fk in table.fks}
        attrs: Dict[str, Any] = {
            "__tablename__": table.name,
            "__table_args__": {"extend_existing": True},
            "__retorm_relationships__": {},
        }
        for col in table.columns:
            args = []
            fk = fk_by_col.get(col.name)
            if fk is not None:
                args.append(ForeignKey(f"{fk.ref_table}.{fk.ref_col}"))
            attrs[col.name] = Column(
                _to_sqla_type(col.col_type),
                *args,
                primary_key=col.is_pk,
                nullable=(False if col.is_pk else col.nullable),
            )
        models[table.name] = type(_model_name(table.name), (base,), attrs)

    for table in schema.tables:
        src_cls = models[table.name]
        for idx, fk in enumerate(table.fks):
            ref_cls = models[fk.ref_table]
            rel_name = _unique_attr_name(src_cls, fk.ref_table.rstrip("s") or fk.ref_table, idx)
            back_name = _unique_attr_name(ref_cls, f"{table.name}_collection", idx)
            fk_attr = getattr(src_cls, fk.src_col)
            ref_pk_attr = getattr(ref_cls, fk.ref_col)
            rel_kwargs = {
                "foreign_keys": [fk_attr],
                "back_populates": back_name,
            }
            if src_cls is ref_cls:
                # Self-referential FKs need remote_side so SQLAlchemy can keep
                # the parent link as many-to-one and the collection side as
                # one-to-many.
                rel_kwargs["remote_side"] = [ref_pk_attr]
            setattr(src_cls, rel_name, relationship(ref_cls, **rel_kwargs))
            setattr(
                ref_cls,
                back_name,
                relationship(
                    src_cls,
                    foreign_keys=[fk_attr],
                    back_populates=rel_name,
                ),
            )
            src_cls.__retorm_relationships__[rel_name] = {
                "source_col": fk.src_col,
                "target_table": fk.ref_table,
                "target_col": fk.ref_col,
                "direction": "many_to_one",
            }
            ref_cls.__retorm_relationships__[back_name] = {
                "source_col": fk.ref_col,
                "target_table": table.name,
                "target_col": fk.src_col,
                "direction": "one_to_many",
            }

    _MODEL_CACHE[signature] = models
    return models


def _unique_attr_name(model_cls, base_name: str, idx: int) -> str:
    name = base_name
    suffix = 1
    while hasattr(model_cls, name):
        suffix += 1
        name = f"{base_name}_{idx}_{suffix}"
    return name


def _schema_signature(schema: Schema) -> Tuple:
    return tuple(
        (
            table.name,
            tuple((col.name, col.col_type.value, col.nullable, col.is_pk) for col in table.columns),
            tuple((fk.src_col, fk.ref_table, fk.ref_col) for fk in table.fks),
        )
        for table in schema.tables
    )


def _model_name(table_name: str) -> str:
    return "".join(part.capitalize() for part in table_name.split("_"))


def _to_sqla_type(col_type: ColType):
    if col_type == ColType.INT:
        return Integer
    if col_type == ColType.FLOAT:
        return Float
    if col_type == ColType.VARCHAR:
        return String(64)
    raise UnsupportedTrueORM(f"unsupported schema column type: {col_type}")
