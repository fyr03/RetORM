"""
generator/ir_gen.py

Random IR generation for RetORM.

Generation order:
  Scan -> optional Join chain -> optional Filter -> optional GroupBy/Aggregate
  -> optional Having -> Project
"""

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from generator.schema_gen import ColType, ForeignKey, Schema, TableSchema
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
    OrderBy,
    OrderKey,
    Or,
    Project,
    QueryNode,
    Scan,
    SelectItem,
    WhenClause,
)


@dataclass
class GenContext:
    visible_cols: List[str] = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    agg_aliases: List[str] = field(default_factory=list)
    group_fields: List[str] = field(default_factory=list)
    has_groupby: bool = False
    query_nullable_cols: set = field(default_factory=set)
    left_join_right_aliases: set = field(default_factory=set)
    projected_fields: List[str] = field(default_factory=list)


def generate_ir(
    schema: Schema,
    join_prob: float = 0.55,
    filter_prob: float = 0.6,
    groupby_prob: float = 0.45,
    having_prob: float = 0.6,
    stress_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[QueryNode, GenContext]:
    stress_mode = stress_mode or "balanced"
    template = _build_stress_template(schema, stress_mode)
    retry_budget = (
        max(1, config.STRESS_RETRY_BUDGET)
        if stress_mode != "balanced"
        else 2
    )

    last_result = None
    for attempt in range(retry_budget):
        if seed is not None:
            random.seed(seed + attempt * 9973)
        join_p, filter_p, groupby_p, having_p = _apply_stress_mode(
            stress_mode, join_prob, filter_prob, groupby_prob, having_prob
        )
        result = _generate_ir_once(
            schema,
            join_p,
            filter_p,
            groupby_p,
            having_p,
            stress_mode,
            template,
        )
        last_result = result
        ir, ctx = result
        if _ir_satisfies_stress_mode(ir, ctx, stress_mode, template):
            return result

    return last_result


def _generate_ir_once(
    schema: Schema,
    join_prob: float,
    filter_prob: float,
    groupby_prob: float,
    having_prob: float,
    stress_mode: str,
    template: Dict[str, object],
) -> Tuple[QueryNode, GenContext]:
    main_table = _choose_main_table(schema, stress_mode, template)
    main_alias = main_table.name[0]
    node: QueryNode = Scan(main_table.name, main_alias)

    ctx = GenContext()
    _add_table_to_ctx(ctx, main_table, main_alias)

    joined_table_names = {main_table.name}
    used_aliases = {main_alias}
    join_step = 0
    target_joins = _target_join_count(schema, stress_mode, template)

    while len(joined_table_names) < len(schema.tables):
        candidates = _find_join_extensions(schema, joined_table_names)
        if not candidates:
            break

        must_take = join_step < target_joins
        step_prob = join_prob if join_step == 0 else _extra_join_prob(stress_mode, join_step)
        if not must_take and random.random() >= step_prob:
            break

        extension = _choose_join_extension(
            candidates,
            stress_mode,
            template=template,
            join_step=join_step,
            target_joins=target_joins,
        )
        existing_alias = _find_alias_for_table(ctx, extension["existing_table"])
        join_table = schema.get_table(extension["new_table"])
        if existing_alias is None or join_table is None:
            break

        join_alias = _make_unique_alias(join_table.name[0], used_aliases)
        want_left_join = (
            bool(template.get("force_left_join"))
            and not ctx.left_join_right_aliases
            and extension["left_join_safe"]
        )
        join_type = _choose_join_type(
            stress_mode,
            allow_left=extension["left_join_safe"],
            force_left=want_left_join,
        )
        on_cond = _make_fk_condition_for_extension(
            extension["fk"],
            existing_alias,
            join_alias,
            extension["existing_table"],
            extension["new_table"],
        )

        node = Join(
            left=node,
            right=Scan(join_table.name, join_alias),
            on=on_cond,
            join_type=join_type,
        )
        _add_table_to_ctx(
            ctx,
            join_table,
            join_alias,
            query_nullable=(join_type == JoinType.LEFT),
            mark_left_join_right=(join_type == JoinType.LEFT),
        )

        joined_table_names.add(join_table.name)
        used_aliases.add(join_alias)
        join_step += 1

    effective_filter_prob = min(1.0, filter_prob + 0.15) if join_step else filter_prob
    should_filter = bool(template.get("require_filter"))
    if not should_filter:
        should_filter = bool(ctx.visible_cols) and random.random() < effective_filter_prob
    if ctx.visible_cols and should_filter:
        cond = _generate_condition(
            ctx,
            schema,
            stress_mode=stress_mode,
            template=template,
        )
        if cond is not None:
            node = Filter(condition=cond, child=node)

    if schema.tables and _should_apply_subquery_condition(stress_mode, template):
        subquery_cond = _generate_subquery_condition(
            ctx,
            schema,
            stress_mode=stress_mode,
            template=template,
        )
        if subquery_cond is not None:
            node = _attach_filter_condition(node, subquery_cond)

    effective_groupby_prob = min(1.0, groupby_prob + 0.2) if join_step else groupby_prob
    should_groupby = bool(template.get("require_groupby"))
    if not should_groupby:
        should_groupby = bool(ctx.visible_cols) and random.random() < effective_groupby_prob
    if ctx.visible_cols and should_groupby:
        numeric_cols = _get_numeric_cols(ctx)
        all_cols = ctx.visible_cols

        group_fields = _choose_group_fields(ctx, stress_mode, all_cols, template=template)
        if not group_fields:
            group_fields = all_cols[:1]

        agg_cols = _get_preferred_agg_cols(ctx, stress_mode, numeric_cols, all_cols)
        if not agg_cols:
            agg_cols = all_cols

        min_aggs = min(max(1, int(template.get("min_aggs", 1))), len(agg_cols))
        max_aggs = min(max(min_aggs, int(template.get("max_aggs", 2))), len(agg_cols))
        num_aggs = min_aggs if min_aggs == max_aggs else random.randint(min_aggs, max_aggs)

        agg_list = []
        used_agg_aliases = set()
        for _ in range(num_aggs):
            agg_col = random.choice(agg_cols)
            agg_func = _choose_agg_func(stress_mode, agg_col, ctx, template=template)
            agg_field = _choose_agg_field(agg_func, agg_col, ctx, template=template)

            base_alias = (
                f"{agg_func.value.lower()}_"
                f"{agg_field.replace('.', '_').replace('*', 'all')}"
            )
            alias_name = base_alias
            suffix = 1
            while alias_name in used_agg_aliases:
                alias_name = f"{base_alias}_{suffix}"
                suffix += 1
            used_agg_aliases.add(alias_name)
            agg_list.append(Aggregate(agg_func, agg_field, alias_name))

        node = GroupBy(fields=group_fields, aggregates=agg_list, child=node)
        ctx.has_groupby = True
        ctx.group_fields = group_fields
        ctx.agg_aliases = [agg.alias for agg in agg_list]

        effective_having_prob = min(1.0, having_prob + 0.1) if join_step else having_prob
        should_having = bool(template.get("require_having"))
        if not should_having:
            should_having = bool(ctx.agg_aliases) and random.random() < effective_having_prob
        if ctx.agg_aliases and should_having:
            having_cond = _generate_having_condition(
                ctx,
                stress_mode=stress_mode,
                template=template,
            )
            if having_cond is not None:
                node = Having(condition=having_cond, child=node)

    project_fields = _choose_project_fields(ctx, stress_mode=stress_mode, template=template)
    project_fields = _maybe_wrap_project_fields(
        project_fields,
        ctx,
        stress_mode=stress_mode,
        template=template,
    )
    if project_fields:
        node = Project(fields=project_fields, child=node)
        ctx.projected_fields = project_fields

    if project_fields and _should_apply_distinct(stress_mode, template):
        node = Distinct(child=node)

    order_keys = _choose_orderby_keys(
        project_fields,
        ctx,
        stress_mode=stress_mode,
        template=template,
    )
    if order_keys:
        node = OrderBy(keys=order_keys, child=node)

    if order_keys and (
        template.get("force_distinct_order_limit")
        or _should_apply_limit_offset(stress_mode, template)
    ):
        limit = random.randint(1, 5)
        offset = random.randint(0, 3)
        node = LimitOffset(limit=limit, offset=offset, child=node)

    return node, ctx


def _build_stress_template(schema: Schema, stress_mode: str) -> Dict[str, object]:
    can_join = bool(schema.fk_pairs())
    can_multi_join = len(schema.tables) >= 3 and len(schema.fk_pairs()) >= 2
    ref_tables = {fk.ref_table for fk in schema.fk_pairs()}

    template: Dict[str, object] = {
        "target_joins": 0,
        "require_filter": False,
        "require_groupby": False,
        "require_having": False,
        "force_left_join": False,
        "force_null_compare": False,
        "force_right_projection": False,
        "force_duplicate_projection": False,
        "require_distinct": False,
        "require_orderby": False,
        "prefer_orderby_agg": False,
        "prefer_sparse_projection": False,
        "min_aggs": 1,
        "max_aggs": 2,
        "count_field_prob": 0.18,
        "prefer_nullable_agg_field": False,
        "ref_tables": ref_tables,
        "require_subquery": False,
        "prefer_exists_subquery": False,
        "force_distinct_order_limit": False,
        "predicate_depth": 2,
        "max_group_fields": 2,
        "max_project_fields": 4,
    }

    if stress_mode == "join_heavy":
        template.update(
            {
                "target_joins": 2 if can_multi_join else (1 if can_join else 0),
                "require_filter": True,
                "force_right_projection": can_join,
                "min_aggs": 1,
                "max_aggs": 2,
                "count_field_prob": 0.25,
                "require_orderby": True,
                "predicate_depth": 3,
            }
        )
    elif stress_mode == "groupby_heavy":
        template.update(
            {
                "target_joins": 1 if can_join and len(schema.tables) >= 2 else 0,
                "require_filter": True,
                "require_groupby": True,
                "require_having": True,
                "min_aggs": 2,
                "max_aggs": 3,
                "count_field_prob": 0.45,
                "prefer_nullable_agg_field": True,
                "require_orderby": True,
                "prefer_orderby_agg": True,
                "predicate_depth": 3,
                "max_group_fields": 3,
                "max_project_fields": 5,
            }
        )
    elif stress_mode == "duplicate_column_heavy":
        template.update(
            {
                "target_joins": 1 if can_join else 0,
                "force_duplicate_projection": True,
                "count_field_prob": 0.22,
                "require_distinct": True,
                "prefer_sparse_projection": True,
            }
        )
    elif stress_mode == "null_heavy":
        template.update(
            {
                "target_joins": 1 if can_join else 0,
                "require_filter": True,
                "force_left_join": bool(ref_tables),
                "force_null_compare": True,
                "force_right_projection": can_join,
                "require_groupby": can_join and len(schema.tables) >= 2,
                "require_having": False,
                "min_aggs": 1,
                "max_aggs": 2,
                "count_field_prob": 0.35,
                "prefer_nullable_agg_field": True,
                "require_orderby": True,
                "predicate_depth": 3,
            }
        )
    elif stress_mode == "orderby_heavy":
        template.update(
            {
                "target_joins": 2 if can_multi_join else (1 if can_join else 0),
                "require_filter": True,
                "require_orderby": True,
                "force_right_projection": can_join,
                "prefer_orderby_agg": True,
                "min_aggs": 1,
                "max_aggs": 2,
                "count_field_prob": 0.4,
                "predicate_depth": 3,
            }
        )
    elif stress_mode == "distinct_heavy":
        template.update(
            {
                "target_joins": 1 if can_join else 0,
                "require_filter": True,
                "require_distinct": True,
                "prefer_sparse_projection": True,
                "force_duplicate_projection": can_join,
                "require_orderby": True,
            }
        )
    elif stress_mode == "subquery_heavy":
        template.update(
            {
                "target_joins": 2 if can_multi_join else (1 if can_join else 0),
                "require_filter": True,
                "require_orderby": True,
                "require_subquery": True,
                "prefer_exists_subquery": True,
                "predicate_depth": 3,
                "min_aggs": 1,
                "max_aggs": 2,
                "max_project_fields": 5,
            }
        )
    elif stress_mode == "combo_heavy":
        template.update(
            {
                "target_joins": 3 if len(schema.tables) >= 4 and len(schema.fk_pairs()) >= 3 else (2 if can_multi_join else (1 if can_join else 0)),
                "require_filter": True,
                "require_groupby": True,
                "require_having": True,
                "require_distinct": True,
                "require_orderby": True,
                "require_subquery": True,
                "force_distinct_order_limit": True,
                "prefer_exists_subquery": False,
                "force_left_join": bool(ref_tables),
                "force_right_projection": can_join,
                "prefer_orderby_agg": True,
                "min_aggs": 2,
                "max_aggs": 3,
                "count_field_prob": 0.4,
                "predicate_depth": 4,
                "max_group_fields": 3,
                "max_project_fields": 6,
            }
        )

    return template


def _ir_satisfies_stress_mode(
    ir: QueryNode,
    ctx: GenContext,
    stress_mode: str,
    template: Dict[str, object],
) -> bool:
    if stress_mode == "balanced":
        return True

    features = _collect_ir_features(ir)

    if features["join_count"] < int(template.get("target_joins", 0)):
        return False
    if template.get("force_left_join") and not features["has_left_join"]:
        return False
    if template.get("require_groupby") and not features["has_groupby"]:
        return False
    if template.get("require_having") and not features["has_having"]:
        return False
    if template.get("force_null_compare") and not features["has_null_predicate"]:
        return False
    if template.get("force_right_projection") and ctx.left_join_right_aliases:
        if not features["has_left_join_right_projection"]:
            return False
    if template.get("force_duplicate_projection") and not features["has_duplicate_projection"]:
        return False
    if template.get("require_distinct") and not features["has_distinct"]:
        return False
    if template.get("require_orderby") and not features["has_orderby"]:
        return False
    if template.get("require_subquery") and not features["has_subquery"]:
        return False
    if template.get("prefer_orderby_agg") and features["has_groupby"] and not features["has_orderby_agg"]:
        return False
    if template.get("force_distinct_order_limit") and not features["has_distinct_order_limit"]:
        return False

    if stress_mode == "join_heavy":
        return features["has_join"] and features["has_orderby"]
    if stress_mode == "groupby_heavy":
        return features["has_groupby"] and features["has_having"] and features["has_orderby_agg"]
    if stress_mode == "duplicate_column_heavy":
        return features["has_duplicate_projection"] and features["has_distinct"]
    if stress_mode == "null_heavy":
        if template.get("force_left_join") and not features["has_left_join"]:
            return False
        return features["has_null_predicate"] and features["has_orderby"]
    if stress_mode == "orderby_heavy":
        return features["has_orderby"]
    if stress_mode == "distinct_heavy":
        return features["has_distinct"]
    if stress_mode == "subquery_heavy":
        return features["has_subquery"] and features["has_orderby"]
    if stress_mode == "combo_heavy":
        return (
            features["has_subquery"]
            and features["has_groupby"]
            and features["has_having"]
            and features["has_distinct_order_limit"]
            and features["join_count"] >= 2
        )
    return True


def _collect_ir_features(node: QueryNode) -> Dict[str, object]:
    features = {
        "join_count": 0,
        "has_join": False,
        "has_left_join": False,
        "has_groupby": False,
        "has_having": False,
        "has_null_predicate": False,
        "has_duplicate_projection": False,
        "has_left_join_right_projection": False,
        "has_distinct": False,
        "has_orderby": False,
        "has_orderby_agg": False,
        "has_limit_offset": False,
        "has_subquery": False,
        "has_exists_subquery": False,
        "has_in_subquery": False,
        "has_distinct_order_limit": False,
    }
    left_join_right_aliases = set()

    def collect_aliases(cur: QueryNode) -> set:
        if isinstance(cur, Scan):
            return {cur.alias}
        if isinstance(cur, Join):
            return collect_aliases(cur.left) | collect_aliases(cur.right)
        if hasattr(cur, "child"):
            return collect_aliases(cur.child)
        return set()

    def field_uses_left_join_right(field_name: str) -> bool:
        return isinstance(field_name, str) and "." in field_name and (
            field_name.split(".", 1)[0] in left_join_right_aliases
        )

    def visit_condition(cond: Condition) -> None:
        if isinstance(cond, Compare):
            if cond.value is None:
                features["has_null_predicate"] = True
            return
        if isinstance(cond, (InList, Between, Like)):
            return
        if isinstance(cond, Exists):
            features["has_subquery"] = True
            features["has_exists_subquery"] = True
            visit(cond.subquery)
            return
        if isinstance(cond, InSubquery):
            features["has_subquery"] = True
            features["has_in_subquery"] = True
            visit(cond.subquery)
            return
        if isinstance(cond, (And, Or)):
            visit_condition(cond.left)
            visit_condition(cond.right)
            return
        if isinstance(cond, Not):
            visit_condition(cond.child)

    def project_field_name(field) -> str:
        if isinstance(field, SelectItem):
            return field.alias
        return str(field)

    def visit(cur: QueryNode) -> None:
        if isinstance(cur, Join):
            features["has_join"] = True
            features["join_count"] += 1
            visit_condition(cur.on)
            if cur.join_type == JoinType.LEFT:
                features["has_left_join"] = True
                left_join_right_aliases.update(collect_aliases(cur.right))
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, Filter):
            visit_condition(cur.condition)
            visit(cur.child)
            return
        if isinstance(cur, GroupBy):
            features["has_groupby"] = True
            visit(cur.child)
            return
        if isinstance(cur, Having):
            features["has_having"] = True
            visit_condition(cur.condition)
            visit(cur.child)
            return
        if isinstance(cur, Project):
            short_names = [_short_field_name(project_field_name(field)) for field in cur.fields]
            features["has_duplicate_projection"] = len(short_names) != len(set(short_names))
            if any(
                field_uses_left_join_right(field) for field in cur.fields if isinstance(field, str)
            ):
                features["has_left_join_right_projection"] = True
            visit(cur.child)
            return
        if isinstance(cur, Distinct):
            features["has_distinct"] = True
            visit(cur.child)
            return
        if isinstance(cur, OrderBy):
            features["has_orderby"] = True
            if any(isinstance(key.field, str) and "." not in key.field for key in cur.keys):
                features["has_orderby_agg"] = True
            visit(cur.child)
            return
        if isinstance(cur, LimitOffset):
            features["has_limit_offset"] = True
            visit(cur.child)

    visit(node)
    features["has_distinct_order_limit"] = (
        features["has_distinct"] and features["has_orderby"] and features["has_limit_offset"]
    )
    return features


def _choose_main_table(
    schema: Schema,
    stress_mode: str,
    template: Dict[str, object],
) -> TableSchema:
    if stress_mode == "null_heavy":
        ref_tables = template.get("ref_tables", set())
        candidates = [table for table in schema.tables if table.name in ref_tables]
        if candidates:
            return random.choice(candidates)

    if stress_mode in ("join_heavy", "groupby_heavy", "duplicate_column_heavy", "subquery_heavy", "combo_heavy"):
        scored = []
        for table in schema.tables:
            degree = 0
            for fk in schema.fk_pairs():
                if fk.src_table == table.name or fk.ref_table == table.name:
                    degree += 1
            scored.append((degree, table))
        best_degree = max(score for score, _ in scored)
        if best_degree > 0:
            candidates = [table for score, table in scored if score == best_degree]
            return random.choice(candidates)

    return random.choice(schema.tables)


def _target_join_count(
    schema: Schema,
    stress_mode: str,
    template: Dict[str, object],
) -> int:
    max_join_chain = max(0, len(schema.tables) - 1)
    target = min(int(template.get("target_joins", 0)), max_join_chain)
    if stress_mode == "join_heavy" and target == 0 and schema.fk_pairs():
        return 1
    return target


def _add_table_to_ctx(
    ctx: GenContext,
    table: TableSchema,
    alias: str,
    query_nullable: bool = False,
    mark_left_join_right: bool = False,
) -> None:
    ctx.tables[alias] = table
    for col in table.columns:
        field_name = f"{alias}.{col.name}"
        ctx.visible_cols.append(field_name)
        if query_nullable:
            ctx.query_nullable_cols.add(field_name)
    if mark_left_join_right:
        ctx.left_join_right_aliases.add(alias)


def _get_numeric_cols(ctx: GenContext) -> List[str]:
    result = []
    for alias, table in ctx.tables.items():
        for col in table.columns:
            if col.col_type in (ColType.INT, ColType.FLOAT):
                result.append(f"{alias}.{col.name}")
    return result


def _get_string_cols(ctx: GenContext) -> List[str]:
    result = []
    for alias, table in ctx.tables.items():
        for col in table.columns:
            if col.col_type == ColType.VARCHAR:
                result.append(f"{alias}.{col.name}")
    return result


def _get_nullable_numeric_cols(ctx: GenContext) -> List[str]:
    result = []
    for alias, table in ctx.tables.items():
        for col in table.columns:
            field_name = f"{alias}.{col.name}"
            if (
                col.col_type in (ColType.INT, ColType.FLOAT)
                and (col.nullable or field_name in ctx.query_nullable_cols)
            ):
                result.append(field_name)
    return result


def _get_nullable_visible_cols(ctx: GenContext) -> List[str]:
    result = []
    for alias, table in ctx.tables.items():
        for col in table.columns:
            field_name = f"{alias}.{col.name}"
            if col.nullable or field_name in ctx.query_nullable_cols:
                result.append(field_name)
    return result


def _get_left_join_right_visible_cols(ctx: GenContext) -> List[str]:
    result = []
    for alias in ctx.left_join_right_aliases:
        table = ctx.tables.get(alias)
        if table is None:
            continue
        for col in table.columns:
            result.append(f"{alias}.{col.name}")
    return result


def _get_left_join_right_numeric_cols(ctx: GenContext) -> List[str]:
    result = []
    for alias in ctx.left_join_right_aliases:
        table = ctx.tables.get(alias)
        if table is None:
            continue
        for col in table.columns:
            if col.col_type in (ColType.INT, ColType.FLOAT):
                result.append(f"{alias}.{col.name}")
    return result


def _find_joinable_fks(schema: Schema, main_table: TableSchema) -> List[ForeignKey]:
    result = []
    for fk in schema.fk_pairs():
        if fk.src_table == main_table.name or fk.ref_table == main_table.name:
            result.append(fk)
    return result


def _find_join_extensions(schema: Schema, joined_table_names: set) -> List[dict]:
    result = []
    for fk in schema.fk_pairs():
        src_joined = fk.src_table in joined_table_names
        ref_joined = fk.ref_table in joined_table_names
        if src_joined == ref_joined:
            continue

        if ref_joined:
            result.append(
                {
                    "fk": fk,
                    "existing_table": fk.ref_table,
                    "new_table": fk.src_table,
                    "left_join_safe": True,
                }
            )
        else:
            result.append(
                {
                    "fk": fk,
                    "existing_table": fk.src_table,
                    "new_table": fk.ref_table,
                    "left_join_safe": False,
                }
            )
    return result


def _make_fk_condition(
    fk: ForeignKey,
    main_alias: str,
    join_alias: str,
    main_table: TableSchema,
    join_table: TableSchema,
) -> Compare:
    if fk.src_table == main_table.name:
        left = f"{main_alias}.{fk.src_col}"
        right = f"{join_alias}.{fk.ref_col}"
    else:
        left = f"{join_alias}.{fk.src_col}"
        right = f"{main_alias}.{fk.ref_col}"
    return Compare(left, CmpOp.EQ, right)


def _make_fk_condition_for_extension(
    fk: ForeignKey,
    existing_alias: str,
    new_alias: str,
    existing_table_name: str,
    new_table_name: str,
) -> Compare:
    if existing_table_name == fk.ref_table and new_table_name == fk.src_table:
        left = f"{new_alias}.{fk.src_col}"
        right = f"{existing_alias}.{fk.ref_col}"
    elif existing_table_name == fk.src_table and new_table_name == fk.ref_table:
        left = f"{existing_alias}.{fk.src_col}"
        right = f"{new_alias}.{fk.ref_col}"
    else:
        raise ValueError(
            f"invalid join extension: {existing_table_name} -> {new_table_name}"
        )
    return Compare(left, CmpOp.EQ, right)


def _find_alias_for_table(ctx: GenContext, table_name: str) -> Optional[str]:
    for alias, table in ctx.tables.items():
        if table.name == table_name:
            return alias
    return None


def _make_unique_alias(base: str, used_aliases: set) -> str:
    alias = base
    suffix = 2
    while alias in used_aliases:
        alias = f"{base}{suffix}"
        suffix += 1
    return alias


def _choose_join_extension(
    candidates: List[dict],
    stress_mode: str,
    template: Optional[Dict[str, object]] = None,
    join_step: int = 0,
    target_joins: int = 0,
) -> dict:
    if template and template.get("force_left_join") and join_step == 0:
        preferred = [c for c in candidates if c["left_join_safe"]]
        if preferred:
            candidates = preferred

    if stress_mode in ("join_heavy", "null_heavy"):
        preferred = [c for c in candidates if c["left_join_safe"]]
        if preferred and random.random() < 0.75:
            candidates = preferred

    if target_joins >= 2 and join_step == 0:
        bridging = [
            c
            for c in candidates
            if sum(
                1
                for other in candidates
                if other["new_table"] != c["new_table"]
            )
            >= 1
        ]
        if bridging:
            candidates = bridging

    return random.choice(candidates)


def _extra_join_prob(stress_mode: str, join_step: int) -> float:
    base = 0.28
    if stress_mode == "join_heavy":
        base = 0.78
    elif stress_mode == "null_heavy":
        base = 0.7
    elif stress_mode == "groupby_heavy":
        base = 0.5
    elif stress_mode == "duplicate_column_heavy":
        base = 0.42
    elif stress_mode == "subquery_heavy":
        base = 0.62
    elif stress_mode == "combo_heavy":
        base = 0.85
    return max(0.15, base - join_step * 0.1)


def _attach_filter_condition(node: QueryNode, cond: Condition) -> QueryNode:
    if isinstance(node, Filter):
        merged = And(node.condition, cond) if random.random() < 0.75 else Or(node.condition, cond)
        return Filter(condition=merged, child=node.child)
    return Filter(condition=cond, child=node)


def _should_apply_subquery_condition(
    stress_mode: str,
    template: Optional[Dict[str, object]] = None,
) -> bool:
    if template and template.get("require_subquery"):
        return True

    prob = 0.05
    if stress_mode == "join_heavy":
        prob = 0.08
    elif stress_mode == "groupby_heavy":
        prob = 0.1
    elif stress_mode == "orderby_heavy":
        prob = 0.12
    elif stress_mode == "subquery_heavy":
        prob = 0.55
    elif stress_mode == "combo_heavy":
        prob = 0.7
    return random.random() < prob


def _generate_subquery_condition(
    ctx: GenContext,
    schema: Schema,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> Optional[Condition]:
    numeric_fields = [field for field in ctx.visible_cols if _field_is_numeric(ctx, field)]
    string_fields = [field for field in ctx.visible_cols if not _field_is_numeric(ctx, field)]

    can_in_subquery = bool(numeric_fields or string_fields)
    prefer_exists = bool(template and template.get("prefer_exists_subquery"))

    if prefer_exists or not can_in_subquery:
        return _make_exists_subquery(schema, stress_mode=stress_mode, template=template)

    if random.random() < 0.45:
        return _make_exists_subquery(schema, stress_mode=stress_mode, template=template)

    use_numeric = bool(numeric_fields) and (not string_fields or random.random() < 0.7)
    target_field = random.choice(numeric_fields if use_numeric else string_fields)
    target_kind = "numeric" if use_numeric else "string"
    subquery = _build_subquery_query(
        schema,
        target_kind=target_kind,
        stress_mode=stress_mode,
        template=template,
    )
    if subquery is None:
        return _make_exists_subquery(schema, stress_mode=stress_mode, template=template)
    return InSubquery(field=target_field, subquery=subquery, negated=(random.random() < 0.2))


def _make_exists_subquery(
    schema: Schema,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> Optional[Condition]:
    subquery = _build_subquery_query(
        schema,
        target_kind=None,
        stress_mode=stress_mode,
        template=template,
    )
    if subquery is None:
        return None
    return Exists(subquery=subquery, negated=(random.random() < 0.18))


def _build_subquery_query(
    schema: Schema,
    target_kind: Optional[str],
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> Optional[QueryNode]:
    base_candidates = _subquery_projection_candidates(schema, target_kind)
    if not base_candidates:
        return None

    table, column = random.choice(base_candidates)
    alias = f"{table.name[0]}s"
    node: QueryNode = Scan(table.name, alias)
    sub_ctx = GenContext()
    _add_table_to_ctx(sub_ctx, table, alias)

    if schema.fk_pairs() and stress_mode in ("join_heavy", "groupby_heavy", "orderby_heavy", "subquery_heavy", "combo_heavy") and random.random() < 0.55:
        candidates = _find_join_extensions(schema, {table.name})
        if candidates:
            extension = _choose_join_extension(
                candidates,
                "join_heavy" if stress_mode == "combo_heavy" else stress_mode,
                template=template,
                join_step=0,
                target_joins=1,
            )
            join_table = schema.get_table(extension["new_table"])
            if join_table is not None:
                join_alias = _make_unique_alias(f"{join_table.name[0]}s", {alias})
                join_type = JoinType.LEFT if extension["left_join_safe"] and random.random() < 0.25 else JoinType.INNER
                node = Join(
                    left=node,
                    right=Scan(join_table.name, join_alias),
                    on=_make_fk_condition_for_extension(
                        extension["fk"],
                        alias,
                        join_alias,
                        extension["existing_table"],
                        extension["new_table"],
                    ),
                    join_type=join_type,
                )
                _add_table_to_ctx(
                    sub_ctx,
                    join_table,
                    join_alias,
                    query_nullable=(join_type == JoinType.LEFT),
                    mark_left_join_right=(join_type == JoinType.LEFT),
                )

    filter_mode = "groupby_heavy" if stress_mode == "combo_heavy" else stress_mode
    if sub_ctx.visible_cols and random.random() < 0.75:
        cond = _generate_condition(
            sub_ctx,
            schema,
            stress_mode=filter_mode,
            template={"predicate_depth": max(1, int((template or {}).get("predicate_depth", 2)) - 1)},
            allow_subquery=False,
        )
        if cond is not None:
            node = Filter(condition=cond, child=node)

    project_field = f"{alias}.{column.name}"
    if stress_mode in ("subquery_heavy", "combo_heavy") and target_kind == "numeric" and random.random() < 0.35:
        agg_alias = f"max_{project_field.replace('.', '_')}"
        node = GroupBy(
            fields=[project_field],
            aggregates=[Aggregate(AggFunc.MAX, project_field, agg_alias)],
            child=node,
        )
        if random.random() < 0.55:
            node = Having(
                condition=Compare(agg_alias, random.choice([CmpOp.GTE, CmpOp.GT, CmpOp.LTE]), random.randint(0, 100)),
                child=node,
            )
        project_field = project_field

    node = Project(fields=[project_field], child=node)

    if random.random() < 0.45:
        node = Distinct(child=node)
    if random.random() < (0.6 if stress_mode in ("subquery_heavy", "combo_heavy") else 0.25):
        node = OrderBy(
            keys=[OrderKey(project_field, descending=random.random() < 0.5)],
            child=node,
        )
        if random.random() < (0.5 if stress_mode in ("subquery_heavy", "combo_heavy") else 0.15):
            node = LimitOffset(limit=random.randint(1, 4), offset=random.randint(0, 1), child=node)

    return node


def _subquery_projection_candidates(
    schema: Schema,
    target_kind: Optional[str],
) -> List[Tuple[TableSchema, object]]:
    candidates: List[Tuple[TableSchema, object]] = []
    for table in schema.tables:
        for column in table.columns:
            if target_kind == "numeric" and column.col_type not in (ColType.INT, ColType.FLOAT):
                continue
            if target_kind == "string" and column.col_type != ColType.VARCHAR:
                continue
            candidates.append((table, column))
    return candidates


def _generate_condition(
    ctx: GenContext,
    schema: Schema,
    depth: int = 0,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
    allow_subquery: bool = True,
) -> Optional[Condition]:
    numeric_cols = _get_numeric_cols(ctx)
    string_cols = _get_string_cols(ctx)
    nullable_numeric_cols = _get_nullable_numeric_cols(ctx)
    nullable_visible_cols = _get_nullable_visible_cols(ctx)
    left_join_right_numeric_cols = _get_left_join_right_numeric_cols(ctx)
    left_join_right_visible_cols = _get_left_join_right_visible_cols(ctx)

    if not numeric_cols and not nullable_visible_cols and not string_cols:
        return None

    if template and template.get("force_null_compare") and nullable_visible_cols:
        forced = _make_compare(
            numeric_cols,
            string_cols,
            nullable_numeric_cols,
            nullable_visible_cols,
            left_join_right_numeric_cols,
            left_join_right_visible_cols,
            stress_mode,
            force_null=True,
        )
        if forced is not None:
            return forced

    max_depth = int((template or {}).get("predicate_depth", 2))
    if depth >= max_depth:
        return _make_compare(
            numeric_cols,
            string_cols,
            nullable_numeric_cols,
            nullable_visible_cols,
            left_join_right_numeric_cols,
            left_join_right_visible_cols,
            stress_mode,
        )

    roll = random.random()
    if string_cols and roll < 0.18:
        cond = _make_like(string_cols)
        if cond is not None:
            return cond
    if numeric_cols and roll < 0.34:
        cond = _make_between(numeric_cols)
        if cond is not None:
            return cond
    if (numeric_cols or string_cols) and roll < 0.5:
        cond = _make_in_list(numeric_cols, string_cols)
        if cond is not None:
            return cond
    if roll < 0.72:
        return _make_compare(
            numeric_cols,
            string_cols,
            nullable_numeric_cols,
            nullable_visible_cols,
            left_join_right_numeric_cols,
            left_join_right_visible_cols,
            stress_mode,
        )
    if roll < 0.78:
        left = _generate_condition(ctx, schema, depth + 1, stress_mode, template=template, allow_subquery=allow_subquery)
        right = _generate_condition(ctx, schema, depth + 1, stress_mode, template=template, allow_subquery=allow_subquery)
        if left and right:
            return And(left, right)
        return left or right
    if roll < 0.94:
        left = _generate_condition(ctx, schema, depth + 1, stress_mode, template=template, allow_subquery=allow_subquery)
        right = _generate_condition(ctx, schema, depth + 1, stress_mode, template=template, allow_subquery=allow_subquery)
        if left and right:
            return Or(left, right)
        return left or right

    if allow_subquery and schema is not None and depth == 0 and random.random() < 0.16:
        subquery_cond = _generate_subquery_condition(
            ctx,
            schema,
            stress_mode=stress_mode,
            template=template,
        )
        if subquery_cond is not None:
            return subquery_cond

    child = _make_compare(
        numeric_cols,
        string_cols,
        nullable_numeric_cols,
        nullable_visible_cols,
        left_join_right_numeric_cols,
        left_join_right_visible_cols,
        stress_mode,
    )
    return Not(child) if child else None


def _make_compare(
    numeric_cols: List[str],
    string_cols: List[str],
    nullable_numeric_cols: List[str],
    nullable_visible_cols: List[str],
    left_join_right_numeric_cols: List[str],
    left_join_right_visible_cols: List[str],
    stress_mode: str,
    force_null: bool = False,
) -> Optional[Compare]:
    if not numeric_cols and not nullable_visible_cols and not string_cols:
        return None

    if force_null and nullable_visible_cols:
        pool = nullable_visible_cols
        if left_join_right_visible_cols:
            pool = left_join_right_visible_cols
        col = random.choice(pool)
        op = random.choice([CmpOp.EQ, CmpOp.NEQ])
        return Compare(col, op, None)

    null_compare_prob = 0.0
    if stress_mode == "null_heavy":
        null_compare_prob = 0.5
    elif stress_mode == "join_heavy" and left_join_right_visible_cols:
        null_compare_prob = 0.2

    if nullable_visible_cols and random.random() < null_compare_prob:
        pool = nullable_visible_cols
        if (
            stress_mode in ("join_heavy", "null_heavy")
            and left_join_right_visible_cols
            and random.random() < 0.8
        ):
            pool = left_join_right_visible_cols
        col = random.choice(pool)
        op = random.choice([CmpOp.EQ, CmpOp.NEQ])
        return Compare(col, op, None)

    if not numeric_cols:
        if string_cols:
            col = random.choice(string_cols)
            op = random.choice([CmpOp.EQ, CmpOp.NEQ])
            return Compare(col, op, _random_string_literal())
        return None

    pool = numeric_cols
    if (
        stress_mode in ("join_heavy", "null_heavy")
        and left_join_right_numeric_cols
        and random.random() < 0.7
    ):
        pool = left_join_right_numeric_cols
    elif stress_mode == "null_heavy" and nullable_numeric_cols and random.random() < 0.8:
        pool = nullable_numeric_cols

    col = random.choice(pool)
    op = random.choice([CmpOp.GT, CmpOp.GTE, CmpOp.LT, CmpOp.LTE, CmpOp.EQ])
    val = random.randint(0, 100)
    return Compare(col, op, val)


def _make_in_list(numeric_cols: List[str], string_cols: List[str]) -> Optional[Condition]:
    choices = []
    if numeric_cols:
        choices.append("numeric")
    if string_cols:
        choices.append("string")
    if not choices:
        return None

    chosen = random.choice(choices)
    if chosen == "numeric":
        field = random.choice(numeric_cols)
        base = random.randint(0, 100)
        values = [base, base + 1, max(0, base - 1)]
    else:
        field = random.choice(string_cols)
        values = [_random_string_literal() for _ in range(3)]
    return InList(field=field, values=values, negated=(random.random() < 0.25))


def _make_between(numeric_cols: List[str]) -> Optional[Condition]:
    if not numeric_cols:
        return None
    field = random.choice(numeric_cols)
    lower = random.randint(0, 60)
    upper = random.randint(lower, 100)
    return Between(field=field, lower=lower, upper=upper, negated=(random.random() < 0.2))


def _make_like(string_cols: List[str]) -> Optional[Condition]:
    if not string_cols:
        return None
    field = random.choice(string_cols)
    pattern = random.choice(["a%", "%a", "%dup%", "_dge", "%z%"])
    return Like(field=field, pattern=pattern, negated=(random.random() < 0.2))


def _generate_having_condition(
    ctx: GenContext,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> Optional[Condition]:
    if not ctx.agg_aliases:
        return None

    if template and template.get("force_null_compare") and random.random() < 0.15:
        alias_name = random.choice(ctx.agg_aliases)
        return Compare(alias_name, random.choice([CmpOp.EQ, CmpOp.NEQ]), None)

    alias_name = random.choice(ctx.agg_aliases)
    if stress_mode in ("join_heavy", "null_heavy") and random.random() < 0.22:
        return Compare(alias_name, random.choice([CmpOp.EQ, CmpOp.NEQ]), None)
    op = random.choice([CmpOp.GT, CmpOp.GTE, CmpOp.LT, CmpOp.LTE])
    val = random.randint(0, 200)
    return Compare(alias_name, op, val)


def _choose_project_fields(
    ctx: GenContext,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> List[object]:
    candidates = ctx.group_fields + ctx.agg_aliases if ctx.has_groupby else ctx.visible_cols
    if not candidates:
        return []
    max_project_fields = max(1, int((template or {}).get("max_project_fields", 4)))

    if ctx.has_groupby:
        chosen = []
        if ctx.group_fields:
            chosen.append(random.choice(ctx.group_fields))
        if ctx.agg_aliases:
            chosen.append(random.choice(ctx.agg_aliases))
            if stress_mode == "groupby_heavy" and len(ctx.agg_aliases) >= 2:
                chosen.append(random.choice(ctx.agg_aliases))
        chosen = _dedupe_keep_order(chosen)
        remaining = [field for field in candidates if field not in chosen]
        if remaining:
            extra_budget = min(max(0, max_project_fields - len(chosen)), len(remaining))
            extra_num = random.randint(0, extra_budget) if extra_budget > 0 else 0
            if extra_num > 0:
                chosen.extend(random.sample(remaining, extra_num))
        return chosen if chosen else random.sample(candidates, random.randint(1, len(candidates)))

    right_fields = [
        field
        for field in candidates
        if "." in field and field.split(".", 1)[0] in ctx.left_join_right_aliases
    ]
    duplicate_fields = _pick_duplicate_short_name_fields(candidates)

    if template and template.get("prefer_sparse_projection"):
        preferred = duplicate_fields or right_fields or candidates
        pick_num = random.randint(1, min(2, len(preferred)))
        chosen = random.sample(preferred, pick_num)
        return _extend_projection(candidates, chosen, extra_limit=0)

    if template and template.get("force_duplicate_projection") and duplicate_fields:
        return _extend_projection(candidates, duplicate_fields, extra_limit=2)

    if template and template.get("force_right_projection") and right_fields:
        chosen = random.sample(right_fields, random.randint(1, len(right_fields)))
        if duplicate_fields:
            chosen = _dedupe_keep_order(duplicate_fields + chosen)
        return _extend_projection(candidates, chosen, extra_limit=2)

    duplicate_prob = 0.45
    if stress_mode == "duplicate_column_heavy":
        duplicate_prob = 0.9
    elif stress_mode == "join_heavy":
        duplicate_prob = 0.6

    if duplicate_fields and random.random() < duplicate_prob:
        return _extend_projection(candidates, duplicate_fields, extra_limit=2)

    if stress_mode in ("join_heavy", "null_heavy") and right_fields and random.random() < 0.7:
        chosen = random.sample(right_fields, random.randint(1, len(right_fields)))
        return _extend_projection(candidates, chosen, extra_limit=2)

    if stress_mode == "null_heavy":
        nullable_fields = [f for f in _get_nullable_visible_cols(ctx) if f in candidates]
        if nullable_fields and random.random() < 0.75:
            chosen = random.sample(nullable_fields, random.randint(1, len(nullable_fields)))
            return _extend_projection(candidates, chosen, extra_limit=2)

    pick_num = random.randint(1, min(len(candidates), max_project_fields))
    return random.sample(candidates, pick_num)


def _maybe_wrap_project_fields(
    fields: List[object],
    ctx: GenContext,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> List[object]:
    if not fields:
        return fields

    wrapped = list(fields)
    numeric_candidates = [
        field
        for field in wrapped
        if isinstance(field, str) and _field_is_numeric(ctx, field)
    ]

    if numeric_candidates and random.random() < 0.22:
        base_field = random.choice(numeric_candidates)
        alias = f"{_short_field_name(base_field)}_calc"
        wrapped.append(
            SelectItem(
                expr=ArithExpr(
                    left=base_field,
                    op=random.choice([ArithOp.ADD, ArithOp.SUB, ArithOp.MUL]),
                    right=random.randint(1, 5),
                ),
                alias=alias,
            )
        )

    case_candidates = [
        field
        for field in wrapped
        if isinstance(field, str) and _field_is_numeric(ctx, field)
    ]
    if case_candidates and random.random() < 0.18:
        base_field = random.choice(case_candidates)
        alias = f"{_short_field_name(base_field)}_bucket"
        wrapped.append(
            SelectItem(
                expr=CaseWhen(
                    cases=[
                        WhenClause(
                            condition=Compare(base_field, CmpOp.GTE, 50),
                            value=1,
                        )
                    ],
                    else_value=0,
                ),
                alias=alias,
            )
        )

    return _dedupe_projection_outputs(wrapped)


def _extend_projection(
    candidates: List[str],
    chosen: List[str],
    extra_limit: int = 2,
) -> List[str]:
    chosen = _dedupe_keep_order(chosen)
    remaining = [field for field in candidates if field not in chosen]
    extra_budget = min(extra_limit, len(remaining))
    extra_num = random.randint(0, extra_budget) if extra_budget > 0 else 0
    if extra_num > 0:
        chosen.extend(random.sample(remaining, extra_num))
    return chosen


def _dedupe_projection_outputs(fields: List[object]) -> List[object]:
    seen = set()
    result = []
    for field in fields:
        name = _projection_output_name(field)
        if name in seen:
            continue
        seen.add(name)
        result.append(field)
    return result


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _projection_output_name(field: object) -> str:
    if isinstance(field, SelectItem):
        return field.alias
    return str(field)


def _projection_order_expr(field: object):
    if isinstance(field, SelectItem):
        return field.expr
    return field


def _field_is_numeric(ctx: GenContext, field_name: str) -> bool:
    if field_name in ctx.agg_aliases:
        return True
    if "." not in field_name:
        return False
    alias, col_name = field_name.split(".", 1)
    table = ctx.tables.get(alias)
    if table is None:
        return False
    column = table.get_column(col_name)
    if column is None:
        return False
    return column.col_type in (ColType.INT, ColType.FLOAT)


def _random_string_literal() -> str:
    return random.choice(["a", "dup", "edge", "", "z"])


def _pick_duplicate_short_name_fields(fields: List[str]) -> List[str]:
    groups = {}
    for field in fields:
        short = field.split(".", 1)[1] if "." in field else field
        groups.setdefault(short, [])
        if field not in groups[short]:
            groups[short].append(field)

    duplicate_groups = [group for group in groups.values() if len(group) >= 2]
    if not duplicate_groups:
        return []

    chosen_group = random.choice(duplicate_groups)
    if len(chosen_group) == 2:
        return chosen_group.copy()
    return random.sample(chosen_group, 2)


def _get_preferred_group_fields(
    ctx: GenContext,
    stress_mode: str,
    fallback_fields: List[str],
) -> List[str]:
    if stress_mode in ("join_heavy", "null_heavy"):
        right_fields = [f for f in _get_left_join_right_visible_cols(ctx) if f in fallback_fields]
        if right_fields and random.random() < 0.7:
            return right_fields

    if stress_mode == "null_heavy":
        nullable_fields = [f for f in _get_nullable_visible_cols(ctx) if f in fallback_fields]
        if nullable_fields:
            return nullable_fields

    return fallback_fields


def _choose_group_fields(
    ctx: GenContext,
    stress_mode: str,
    fallback_fields: List[str],
    template: Optional[Dict[str, object]] = None,
) -> List[str]:
    group_pool = _get_preferred_group_fields(ctx, stress_mode, fallback_fields)
    if not group_pool:
        group_pool = fallback_fields
    if not group_pool:
        return []

    max_group_fields = max(1, int((template or {}).get("max_group_fields", 2)))

    if template and template.get("require_groupby") and stress_mode in ("groupby_heavy", "combo_heavy"):
        pick_num = min(max_group_fields, len(group_pool))
        return random.sample(group_pool, pick_num)

    pick_num = random.randint(1, min(max_group_fields, len(group_pool)))
    return random.sample(group_pool, pick_num)


def _get_preferred_agg_cols(
    ctx: GenContext,
    stress_mode: str,
    numeric_cols: List[str],
    fallback_fields: List[str],
) -> List[str]:
    if stress_mode in ("join_heavy", "null_heavy"):
        right_numeric_cols = _get_left_join_right_numeric_cols(ctx)
        if right_numeric_cols and random.random() < 0.75:
            return right_numeric_cols
    return numeric_cols if numeric_cols else fallback_fields


def _choose_agg_func(
    stress_mode: str,
    agg_col: str,
    ctx: GenContext,
    template: Optional[Dict[str, object]] = None,
) -> AggFunc:
    if (
        stress_mode in ("join_heavy", "null_heavy")
        and "." in agg_col
        and agg_col.split(".", 1)[0] in ctx.left_join_right_aliases
        and random.random() < 0.75
    ):
        return random.choice([AggFunc.SUM, AggFunc.AVG, AggFunc.MAX, AggFunc.MIN, AggFunc.COUNT])

    if template and template.get("require_groupby") and random.random() < 0.35:
        return random.choice([AggFunc.COUNT, AggFunc.AVG, AggFunc.MAX, AggFunc.MIN])
    if stress_mode == "combo_heavy" and random.random() < 0.45:
        return random.choice([AggFunc.COUNT, AggFunc.SUM, AggFunc.AVG, AggFunc.MAX])

    return random.choice(list(AggFunc))


def _choose_agg_field(
    agg_func: AggFunc,
    agg_col: str,
    ctx: GenContext,
    template: Optional[Dict[str, object]] = None,
) -> str:
    if agg_func != AggFunc.COUNT:
        return agg_col

    count_field_prob = 0.18
    if template is not None:
        count_field_prob = float(template.get("count_field_prob", count_field_prob))

    nullable_cols = _get_nullable_visible_cols(ctx)
    if template and template.get("prefer_nullable_agg_field") and nullable_cols and random.random() < 0.7:
        return random.choice(nullable_cols)
    if random.random() < count_field_prob:
        return agg_col
    return "*"


def _choose_join_type(
    stress_mode: str,
    allow_left: bool = True,
    force_left: bool = False,
) -> JoinType:
    if not allow_left:
        return JoinType.INNER
    if force_left:
        return JoinType.LEFT

    left_prob = 0.1
    if stress_mode == "join_heavy":
        left_prob = 0.35
    elif stress_mode == "null_heavy":
        left_prob = 0.6
    elif stress_mode == "groupby_heavy":
        left_prob = 0.2
    return JoinType.LEFT if random.random() < left_prob else JoinType.INNER


def _should_apply_distinct(
    stress_mode: str,
    template: Optional[Dict[str, object]] = None,
) -> bool:
    if template and template.get("require_distinct"):
        return True

    distinct_prob = 0.08
    if stress_mode == "join_heavy":
        distinct_prob = 0.12
    elif stress_mode == "groupby_heavy":
        distinct_prob = 0.14
    elif stress_mode == "duplicate_column_heavy":
        distinct_prob = 0.45
    elif stress_mode == "null_heavy":
        distinct_prob = 0.12
    elif stress_mode == "orderby_heavy":
        distinct_prob = 0.18
    elif stress_mode == "distinct_heavy":
        distinct_prob = 0.9
    elif stress_mode == "subquery_heavy":
        distinct_prob = 0.2
    elif stress_mode == "combo_heavy":
        distinct_prob = 0.85

    return random.random() < distinct_prob


def _should_apply_limit_offset(
    stress_mode: str,
    template: Optional[Dict[str, object]] = None,
) -> bool:
    limit_offset_prob = 0.06
    if stress_mode == "join_heavy":
        limit_offset_prob = 0.1
    elif stress_mode == "groupby_heavy":
        limit_offset_prob = 0.12
    elif stress_mode == "duplicate_column_heavy":
        limit_offset_prob = 0.08
    elif stress_mode == "null_heavy":
        limit_offset_prob = 0.12
    elif stress_mode == "orderby_heavy":
        limit_offset_prob = 0.38
    elif stress_mode == "distinct_heavy":
        limit_offset_prob = 0.18
    elif stress_mode == "subquery_heavy":
        limit_offset_prob = 0.2
    elif stress_mode == "combo_heavy":
        limit_offset_prob = 0.75

    if template and template.get("require_orderby"):
        limit_offset_prob = max(limit_offset_prob, 0.14)

    return random.random() < limit_offset_prob


def _choose_orderby_keys(
    project_fields: List[object],
    ctx: GenContext,
    stress_mode: str = "balanced",
    template: Optional[Dict[str, object]] = None,
) -> List[OrderKey]:
    if not project_fields:
        return []

    if template and template.get("require_orderby"):
        should_orderby = True
    else:
        orderby_prob = 0.1
        if stress_mode == "join_heavy":
            orderby_prob = 0.18
        elif stress_mode == "groupby_heavy":
            orderby_prob = 0.24
        elif stress_mode == "duplicate_column_heavy":
            orderby_prob = 0.16
        elif stress_mode == "null_heavy":
            orderby_prob = 0.22
        elif stress_mode == "orderby_heavy":
            orderby_prob = 0.95
        elif stress_mode == "distinct_heavy":
            orderby_prob = 0.65
        elif stress_mode == "subquery_heavy":
            orderby_prob = 0.72
        elif stress_mode == "combo_heavy":
            orderby_prob = 0.95
        should_orderby = random.random() < orderby_prob

    if not should_orderby:
        return []

    order_exprs = [_projection_order_expr(field) for field in project_fields]
    unique_fields = []
    seen_exprs = set()
    for expr in order_exprs:
        key = repr(expr)
        if key in seen_exprs:
            continue
        seen_exprs.add(key)
        unique_fields.append(expr)
    ordered_fields: List[str] = []

    if template and template.get("prefer_orderby_agg"):
        agg_fields = [
            field
            for field in unique_fields
            if isinstance(field, str) and field in ctx.agg_aliases
        ]
        other_fields = [field for field in unique_fields if field not in agg_fields]
        ordered_fields.extend(agg_fields)
        ordered_fields.extend(other_fields)
    elif stress_mode in ("join_heavy", "null_heavy", "orderby_heavy"):
        right_fields = [
            field
            for field in unique_fields
            if isinstance(field, str) and "." in field and field.split(".", 1)[0] in ctx.left_join_right_aliases
        ]
        other_fields = [field for field in unique_fields if field not in right_fields]
        ordered_fields.extend(right_fields)
        ordered_fields.extend(other_fields)
    else:
        ordered_fields = unique_fields

    if not ordered_fields:
        return []

    return [
        OrderKey(field=field, descending=random.random() < 0.4)
        for field in ordered_fields
    ]


def _apply_stress_mode(
    stress_mode: str,
    join_prob: float,
    filter_prob: float,
    groupby_prob: float,
    having_prob: float,
) -> Tuple[float, float, float, float]:
    if stress_mode == "join_heavy":
        return (
            min(1.0, join_prob + 0.28),
            min(1.0, filter_prob + 0.14),
            min(1.0, groupby_prob + 0.12),
            min(1.0, having_prob + 0.1),
        )
    if stress_mode == "duplicate_column_heavy":
        return (
            min(1.0, join_prob + 0.22),
            filter_prob,
            min(1.0, groupby_prob + 0.08),
            having_prob,
        )
    if stress_mode == "null_heavy":
        return (
            min(1.0, join_prob + 0.1),
            min(1.0, filter_prob + 0.14),
            min(1.0, groupby_prob + 0.18),
            min(1.0, having_prob + 0.1),
        )
    if stress_mode == "groupby_heavy":
        return (
            min(1.0, join_prob + 0.05),
            min(1.0, filter_prob + 0.08),
            min(1.0, groupby_prob + 0.35),
            min(1.0, having_prob + 0.28),
        )
    if stress_mode == "orderby_heavy":
        return (
            min(1.0, join_prob + 0.2),
            min(1.0, filter_prob + 0.12),
            min(1.0, groupby_prob + 0.18),
            min(1.0, having_prob + 0.1),
        )
    if stress_mode == "distinct_heavy":
        return (
            min(1.0, join_prob + 0.12),
            min(1.0, filter_prob + 0.12),
            min(1.0, groupby_prob + 0.05),
            having_prob,
        )
    if stress_mode == "subquery_heavy":
        return (
            min(1.0, join_prob + 0.18),
            min(1.0, filter_prob + 0.2),
            min(1.0, groupby_prob + 0.12),
            min(1.0, having_prob + 0.08),
        )
    if stress_mode == "combo_heavy":
        return (
            min(1.0, join_prob + 0.28),
            min(1.0, filter_prob + 0.22),
            min(1.0, groupby_prob + 0.28),
            min(1.0, having_prob + 0.22),
        )
    return join_prob, filter_prob, groupby_prob, having_prob


def _short_field_name(field_name: str) -> str:
    return field_name.split(".", 1)[1] if "." in field_name else field_name


if __name__ == "__main__":
    from generator.schema_gen import generate_schema, print_schema
    from ir.nodes import pretty_print

    schema = generate_schema(num_tables=4, cols_per_table=3, fk_prob=0.8, seed=42)
    print_schema(schema)
    print()
    for mode in (
        "balanced",
        "join_heavy",
        "groupby_heavy",
        "duplicate_column_heavy",
        "null_heavy",
        "orderby_heavy",
        "distinct_heavy",
        "subquery_heavy",
        "combo_heavy",
    ):
        ir, ctx = generate_ir(schema, stress_mode=mode, seed=7)
        print(f"--- mode={mode} ---")
        print(pretty_print(ir))
        print(f"visible_cols={ctx.visible_cols}")
        print(f"left_join_right_aliases={sorted(ctx.left_join_right_aliases)}")
        print()
