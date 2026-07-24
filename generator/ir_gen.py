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
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.schema_gen import Column, ColType, ForeignKey, Schema, TableSchema
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


@dataclass
class GenContext:
    visible_cols: List[str] = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    agg_aliases: List[str] = field(default_factory=list)
    group_fields: List[str] = field(default_factory=list)
    has_groupby: bool = False
    query_nullable_cols: set = field(default_factory=set)
    left_join_right_aliases: set = field(default_factory=set)


def generate_ir(
    schema: Schema,
    join_prob: float = 0.55,
    filter_prob: float = 0.6,
    groupby_prob: float = 0.45,
    having_prob: float = 0.6,
    stress_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[QueryNode, GenContext]:
    if seed is not None:
        random.seed(seed)
    stress_mode = stress_mode or "balanced"

    join_prob, filter_prob, groupby_prob, having_prob = _apply_stress_mode(
        stress_mode, join_prob, filter_prob, groupby_prob, having_prob
    )

    main_table = random.choice(schema.tables)
    main_alias = main_table.name[0]
    node: QueryNode = Scan(main_table.name, main_alias)

    ctx = GenContext()
    _add_table_to_ctx(ctx, main_table, main_alias)

    joined = False
    joined_table_names = {main_table.name}
    used_aliases = {main_alias}
    join_step = 0

    while len(joined_table_names) < len(schema.tables):
        candidates = _find_join_extensions(schema, joined_table_names)
        if not candidates:
            break

        if join_step == 0:
            if random.random() >= join_prob:
                break
        elif random.random() >= _extra_join_prob(stress_mode, join_step):
            break

        extension = _choose_join_extension(candidates, stress_mode)
        existing_alias = _find_alias_for_table(ctx, extension["existing_table"])
        join_table = schema.get_table(extension["new_table"])
        if existing_alias is None or join_table is None:
            break

        join_alias = _make_unique_alias(join_table.name[0], used_aliases)
        join_type = _choose_join_type(
            stress_mode,
            allow_left=extension["left_join_safe"],
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

        joined = True
        joined_table_names.add(join_table.name)
        used_aliases.add(join_alias)
        join_step += 1

    effective_filter_prob = min(1.0, filter_prob + 0.15) if joined else filter_prob
    if ctx.visible_cols and random.random() < effective_filter_prob:
        cond = _generate_condition(ctx, schema, stress_mode=stress_mode)
        if cond is not None:
            node = Filter(condition=cond, child=node)

    effective_groupby_prob = min(1.0, groupby_prob + 0.2) if joined else groupby_prob
    if ctx.visible_cols and random.random() < effective_groupby_prob:
        numeric_cols = _get_numeric_cols(ctx)
        all_cols = ctx.visible_cols

        group_fields = _choose_group_fields(ctx, stress_mode, all_cols)
        if not group_fields:
            group_fields = all_cols[:1]

        agg_cols = _get_preferred_agg_cols(ctx, stress_mode, numeric_cols, all_cols)
        num_aggs = random.randint(1, min(2, len(agg_cols)))
        agg_list = []
        used_agg_aliases = set()

        for _ in range(num_aggs):
            agg_col = random.choice(agg_cols)
            agg_func = _choose_agg_func(stress_mode, agg_col, ctx)
            agg_field = "*" if agg_func == AggFunc.COUNT else agg_col

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

        effective_having_prob = min(1.0, having_prob + 0.1) if joined else having_prob
        if ctx.agg_aliases and random.random() < effective_having_prob:
            having_cond = _generate_having_condition(ctx, stress_mode=stress_mode)
            if having_cond is not None:
                node = Having(condition=having_cond, child=node)

    project_fields = _choose_project_fields(ctx, stress_mode=stress_mode)
    if project_fields:
        node = Project(fields=project_fields, child=node)

    return node, ctx


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


def _choose_join_extension(candidates: List[dict], stress_mode: str) -> dict:
    if stress_mode in ("join_heavy", "null_heavy"):
        preferred = [c for c in candidates if c["left_join_safe"]]
        if preferred and random.random() < 0.75:
            candidates = preferred
    return random.choice(candidates)


def _extra_join_prob(stress_mode: str, join_step: int) -> float:
    base = 0.28
    if stress_mode == "join_heavy":
        base = 0.72
    elif stress_mode == "null_heavy":
        base = 0.62
    elif stress_mode == "groupby_heavy":
        base = 0.48
    elif stress_mode == "duplicate_column_heavy":
        base = 0.4
    return max(0.15, base - join_step * 0.12)


def _generate_condition(
    ctx: GenContext,
    schema: Schema,
    depth: int = 0,
    stress_mode: str = "balanced",
) -> Optional[Condition]:
    numeric_cols = _get_numeric_cols(ctx)
    nullable_numeric_cols = _get_nullable_numeric_cols(ctx)
    nullable_visible_cols = _get_nullable_visible_cols(ctx)
    left_join_right_numeric_cols = _get_left_join_right_numeric_cols(ctx)
    left_join_right_visible_cols = _get_left_join_right_visible_cols(ctx)

    if not numeric_cols and not nullable_visible_cols:
        return None

    if depth >= 2:
        return _make_compare(
            numeric_cols,
            nullable_numeric_cols,
            nullable_visible_cols,
            left_join_right_numeric_cols,
            left_join_right_visible_cols,
            stress_mode,
        )

    roll = random.random()
    if roll < 0.6:
        return _make_compare(
            numeric_cols,
            nullable_numeric_cols,
            nullable_visible_cols,
            left_join_right_numeric_cols,
            left_join_right_visible_cols,
            stress_mode,
        )
    if roll < 0.8:
        left = _generate_condition(ctx, schema, depth + 1, stress_mode)
        right = _generate_condition(ctx, schema, depth + 1, stress_mode)
        if left and right:
            return And(left, right)
        return left or right
    if roll < 0.95:
        left = _generate_condition(ctx, schema, depth + 1, stress_mode)
        right = _generate_condition(ctx, schema, depth + 1, stress_mode)
        if left and right:
            return Or(left, right)
        return left or right

    child = _make_compare(
        numeric_cols,
        nullable_numeric_cols,
        nullable_visible_cols,
        left_join_right_numeric_cols,
        left_join_right_visible_cols,
        stress_mode,
    )
    return Not(child) if child else None


def _make_compare(
    numeric_cols: List[str],
    nullable_numeric_cols: List[str],
    nullable_visible_cols: List[str],
    left_join_right_numeric_cols: List[str],
    left_join_right_visible_cols: List[str],
    stress_mode: str,
) -> Optional[Compare]:
    if not numeric_cols and not nullable_visible_cols:
        return None

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
    val = random.randint(1, 100)
    return Compare(col, op, val)


def _generate_having_condition(
    ctx: GenContext,
    stress_mode: str = "balanced",
) -> Optional[Condition]:
    if not ctx.agg_aliases:
        return None
    alias_name = random.choice(ctx.agg_aliases)
    if stress_mode in ("join_heavy", "null_heavy") and random.random() < 0.22:
        return Compare(alias_name, random.choice([CmpOp.EQ, CmpOp.NEQ]), None)
    op = random.choice([CmpOp.GT, CmpOp.GTE, CmpOp.LT, CmpOp.LTE])
    val = random.randint(1, 200)
    return Compare(alias_name, op, val)


def _choose_project_fields(ctx: GenContext, stress_mode: str = "balanced") -> List[str]:
    candidates = ctx.group_fields + ctx.agg_aliases if ctx.has_groupby else ctx.visible_cols
    if not candidates:
        return []

    if not ctx.has_groupby:
        right_fields = [
            field
            for field in candidates
            if "." in field and field.split(".", 1)[0] in ctx.left_join_right_aliases
        ]

        duplicate_fields = _pick_duplicate_short_name_fields(candidates)
        duplicate_prob = 0.45
        if stress_mode == "duplicate_column_heavy":
            duplicate_prob = 0.9
        elif stress_mode == "join_heavy":
            duplicate_prob = 0.6

        if duplicate_fields and random.random() < duplicate_prob:
            chosen = list(duplicate_fields)
            remaining = [field for field in candidates if field not in chosen]
            extra_budget = min(2, len(remaining))
            extra_num = random.randint(0, extra_budget) if extra_budget > 0 else 0
            if extra_num > 0:
                chosen.extend(random.sample(remaining, extra_num))
            return chosen

        if (
            stress_mode in ("join_heavy", "null_heavy")
            and right_fields
            and random.random() < 0.7
        ):
            pick_num = random.randint(1, len(right_fields))
            chosen = random.sample(right_fields, pick_num)
            remaining = [field for field in candidates if field not in chosen]
            if remaining:
                extra_budget = min(2, len(remaining))
                extra_num = random.randint(0, extra_budget) if extra_budget > 0 else 0
                if extra_num > 0:
                    chosen.extend(random.sample(remaining, extra_num))
            return chosen

        if stress_mode == "null_heavy":
            nullable_fields = [f for f in _get_nullable_visible_cols(ctx) if f in candidates]
            if nullable_fields and random.random() < 0.75:
                nullable_num = random.randint(1, len(nullable_fields))
                chosen = random.sample(nullable_fields, nullable_num)
                remaining = [field for field in candidates if field not in chosen]
                if remaining and len(chosen) < len(candidates):
                    extra_budget = min(2, len(remaining))
                    extra_num = random.randint(0, extra_budget) if extra_budget > 0 else 0
                    if extra_num > 0:
                        chosen.extend(random.sample(remaining, extra_num))
                return chosen

    pick_num = random.randint(1, len(candidates))
    return random.sample(candidates, pick_num)


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
) -> List[str]:
    group_pool = _get_preferred_group_fields(ctx, stress_mode, fallback_fields)
    if not group_pool:
        group_pool = fallback_fields
    if not group_pool:
        return []

    pick_num = random.randint(1, min(2, len(group_pool)))
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


def _choose_agg_func(stress_mode: str, agg_col: str, ctx: GenContext) -> AggFunc:
    if (
        stress_mode in ("join_heavy", "null_heavy")
        and "." in agg_col
        and agg_col.split(".", 1)[0] in ctx.left_join_right_aliases
        and random.random() < 0.8
    ):
        return random.choice([AggFunc.SUM, AggFunc.AVG, AggFunc.MAX, AggFunc.MIN])
    return random.choice(list(AggFunc))


def _choose_join_type(stress_mode: str, allow_left: bool = True) -> JoinType:
    if not allow_left:
        return JoinType.INNER

    left_prob = 0.1
    if stress_mode == "join_heavy":
        left_prob = 0.35
    elif stress_mode == "null_heavy":
        left_prob = 0.55
    elif stress_mode == "groupby_heavy":
        left_prob = 0.2
    return JoinType.LEFT if random.random() < left_prob else JoinType.INNER


def _apply_stress_mode(
    stress_mode: str,
    join_prob: float,
    filter_prob: float,
    groupby_prob: float,
    having_prob: float,
) -> Tuple[float, float, float, float]:
    if stress_mode == "join_heavy":
        return (
            min(1.0, join_prob + 0.25),
            min(1.0, filter_prob + 0.1),
            min(1.0, groupby_prob + 0.1),
            min(1.0, having_prob + 0.1),
        )
    if stress_mode == "duplicate_column_heavy":
        return (
            min(1.0, join_prob + 0.2),
            filter_prob,
            min(1.0, groupby_prob + 0.05),
            having_prob,
        )
    if stress_mode == "null_heavy":
        return (
            min(1.0, join_prob + 0.05),
            min(1.0, filter_prob + 0.1),
            min(1.0, groupby_prob + 0.15),
            min(1.0, having_prob + 0.1),
        )
    if stress_mode == "groupby_heavy":
        return (
            join_prob,
            filter_prob,
            min(1.0, groupby_prob + 0.3),
            min(1.0, having_prob + 0.2),
        )
    return join_prob, filter_prob, groupby_prob, having_prob


if __name__ == "__main__":
    from generator.schema_gen import generate_schema, print_schema
    from ir.nodes import pretty_print

    schema = generate_schema(num_tables=4, cols_per_table=3, fk_prob=0.8, seed=42)
    print_schema(schema)
    print()
    for i in range(3):
        ir, ctx = generate_ir(schema, stress_mode="join_heavy", seed=i)
        print(f"--- IR #{i + 1} ---")
        print(pretty_print(ir))
        print(f"visible_cols={ctx.visible_cols}")
        print(f"left_join_right_aliases={sorted(ctx.left_join_right_aliases)}")
        print()
