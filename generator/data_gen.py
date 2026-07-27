"""
generator/data_gen.py

Generate database contents for RetORM.

The generator intentionally mixes:
  - core rows that keep queries non-empty
  - fixed edge rows with boundary values
  - adversarial rows that stress JOIN / NULL / GROUP BY behavior
  - extra noise rows that increase result-set complexity
"""

import os
import random
import string
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from generator.schema_gen import ColType, Column, Schema, TableSchema
from ir.nodes import (
    AggFunc,
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
TableData = Dict[str, List[Row]]


def generate_and_insert(
    schema: Schema,
    ir: QueryNode,
    rows_per_table: int = 8,
    use_z3: bool = True,
    z3_timeout: int = 5,
    stress_mode: str = "balanced",
    seed: Optional[int] = None,
) -> TableData:
    if seed is not None:
        random.seed(seed)

    profile = _extract_generation_profile(ir)

    if use_z3:
        try:
            data = _generate_with_z3(
                schema,
                ir,
                rows_per_table,
                z3_timeout,
                stress_mode=stress_mode,
                profile=profile,
            )
            if data:
                _insert_all(schema, data)
                return data
        except Exception as exc:
            print(f"[data_gen] Z3 generation failed ({exc}), fallback to random")

    data = _generate_random(
        schema,
        rows_per_table,
        stress_mode=stress_mode,
        profile=profile,
    )
    _insert_all(schema, data)
    return data


def generate_random_only(
    schema: Schema,
    rows_per_table: int = 8,
    stress_mode: str = "balanced",
    seed: Optional[int] = None,
) -> TableData:
    if seed is not None:
        random.seed(seed)
    data = _generate_random(
        schema,
        rows_per_table,
        stress_mode=stress_mode,
        profile=None,
    )
    _insert_all(schema, data)
    return data


def _generate_random(
    schema: Schema,
    rows_per_table: int,
    stress_mode: str = "balanced",
    profile: Optional[dict] = None,
) -> TableData:
    data: TableData = {}

    from generator.schema_gen import generate_drop_sqls

    ordered_names = list(reversed(generate_drop_sqls(schema)))
    for table_name in ordered_names:
        table = schema.get_table(table_name)
        rows = _generate_table_rows(
            table,
            rows_per_table,
            existing_data=data,
            stress_mode=stress_mode,
            profile=profile,
        )
        data[table_name] = rows
    return data


def _generate_table_rows(
    table: TableSchema,
    base_rows: int,
    existing_data: TableData,
    stress_mode: str = "balanced",
    profile: Optional[dict] = None,
    first_row_override: Optional[Row] = None,
) -> List[Row]:
    counts = _plan_row_budget(base_rows, stress_mode, table.name, profile)
    pools = {
        col.name: _build_value_pool(col, stress_mode, profile)
        for col in table.columns
        if not col.is_pk and not _is_fk_col(col.name, table)
    }

    rows: List[Row] = []
    next_id = 1

    if first_row_override is not None:
        row = _make_random_row(
            table,
            row_id=next_id,
            existing_data=existing_data,
            stress_mode=stress_mode,
            profile=profile,
            pools=pools,
        )
        row.update(first_row_override)
        row["id"] = next_id
        rows.append(row)
        next_id += 1
        counts["core"] = max(0, counts["core"] - 1)

    for _ in range(counts["core"]):
        rows.append(
            _make_random_row(
                table,
                row_id=next_id,
                existing_data=existing_data,
                stress_mode=stress_mode,
                profile=profile,
                pools=pools,
            )
        )
        next_id += 1

    for edge_index in range(counts["edge"]):
        rows.append(
            _make_edge_row(
                table,
                row_id=next_id,
                existing_data=existing_data,
                stress_mode=stress_mode,
                profile=profile,
                edge_index=edge_index,
            )
        )
        next_id += 1

    for adv_index in range(counts["adversarial"]):
        rows.append(
            _make_adversarial_row(
                table,
                row_id=next_id,
                existing_data=existing_data,
                stress_mode=stress_mode,
                profile=profile,
                adv_index=adv_index,
                pools=pools,
            )
        )
        next_id += 1

    for _ in range(counts["noise"]):
        rows.append(
            _make_random_row(
                table,
                row_id=next_id,
                existing_data=existing_data,
                stress_mode=stress_mode,
                profile=profile,
                pools=pools,
                row_kind="noise",
            )
        )
        next_id += 1

    return rows


def _plan_row_budget(
    base_rows: int,
    stress_mode: str,
    table_name: str,
    profile: Optional[dict],
) -> Dict[str, int]:
    edge_rows = config.EDGE_ROWS
    adversarial_rows = config.ADVERSARIAL_ROWS
    extra_random_rows = config.EXTRA_RANDOM_ROWS

    if stress_mode in ("join_heavy", "null_heavy"):
        adversarial_rows += 2
        extra_random_rows += 2
    elif stress_mode == "groupby_heavy":
        adversarial_rows += 1
        edge_rows += 1

    if profile and table_name in profile["left_join_right_tables"]:
        adversarial_rows += 2

    return {
        "core": max(1, base_rows),
        "edge": edge_rows,
        "adversarial": adversarial_rows,
        "noise": extra_random_rows,
    }


def _make_random_row(
    table: TableSchema,
    row_id: int,
    existing_data: TableData,
    stress_mode: str,
    profile: Optional[dict],
    pools: Dict[str, List[Any]],
    row_kind: str = "core",
) -> Row:
    row: Row = {}
    for col in table.columns:
        if col.is_pk:
            row[col.name] = row_id
        elif _is_fk_col(col.name, table):
            row[col.name] = _choose_fk_value(
                col,
                table,
                row_id,
                existing_data,
                stress_mode,
                profile,
                row_kind=row_kind,
            )
        else:
            hotspot_values = _hot_values_for_column(col, profile)
            row[col.name] = _random_value(
                col,
                stress_mode=stress_mode,
                pool=pools.get(col.name),
                row_kind=row_kind,
                hotspot_values=hotspot_values,
            )
    return row


def _make_edge_row(
    table: TableSchema,
    row_id: int,
    existing_data: TableData,
    stress_mode: str,
    profile: Optional[dict],
    edge_index: int,
) -> Row:
    row: Row = {}
    for col in table.columns:
        if col.is_pk:
            row[col.name] = row_id
            continue
        if _is_fk_col(col.name, table):
            row[col.name] = _choose_fk_value(
                col,
                table,
                row_id,
                existing_data,
                stress_mode,
                profile,
                row_kind="edge",
            )
            continue

        if col.nullable and edge_index % 4 == 0:
            row[col.name] = None
            continue

        edge_values = _edge_values_for_column(col, profile)
        row[col.name] = edge_values[edge_index % len(edge_values)]
    return row


def _make_adversarial_row(
    table: TableSchema,
    row_id: int,
    existing_data: TableData,
    stress_mode: str,
    profile: Optional[dict],
    adv_index: int,
    pools: Dict[str, List[Any]],
) -> Row:
    row = _make_random_row(
        table,
        row_id=row_id,
        existing_data=existing_data,
        stress_mode=stress_mode,
        profile=profile,
        pools=pools,
        row_kind="adversarial",
    )

    hot_short_names = profile["hot_short_names"] if profile else set()
    nullable_short_names = profile["nullable_short_names"] if profile else set()

    for col in table.columns:
        if col.is_pk or _is_fk_col(col.name, table):
            continue

        if col.name in nullable_short_names and col.nullable and adv_index % 2 == 0:
            row[col.name] = None
            continue

        if col.name in hot_short_names:
            hot_values = _hot_values_for_column(col, profile)
            if hot_values:
                row[col.name] = hot_values[adv_index % len(hot_values)]
                continue

        if col.col_type == ColType.VARCHAR:
            row[col.name] = ["dup", "", "edge", "dup"][adv_index % 4]

    return row


def _choose_fk_value(
    col: Column,
    table: TableSchema,
    row_id: int,
    existing_data: TableData,
    stress_mode: str,
    profile: Optional[dict],
    row_kind: str,
) -> int:
    ref_table_name = _get_fk_ref_table(col.name, table)
    parent_rows = existing_data.get(ref_table_name or "", [])
    parent_ids = [row["id"] for row in parent_rows]

    if not parent_ids:
        return row_id

    allow_orphan = (
        row_kind in ("adversarial", "edge")
        and profile is not None
        and table.name in profile["left_join_right_tables"]
        and stress_mode in ("join_heavy", "null_heavy", "groupby_heavy")
    )

    if allow_orphan and random.random() < 0.55:
        return max(parent_ids) + row_id + 17

    if row_kind in ("core", "adversarial") and stress_mode in (
        "join_heavy",
        "groupby_heavy",
        "duplicate_column_heavy",
        "null_heavy",
    ):
        hot_ids = parent_ids[: min(3, len(parent_ids))]
        if hot_ids and random.random() < 0.8:
            return random.choice(hot_ids)

    if row_kind == "edge":
        return parent_ids[(row_id - 1) % len(parent_ids)]

    return random.choice(parent_ids)


def _random_value(
    col: Column,
    stress_mode: str = "balanced",
    pool: Optional[List[Any]] = None,
    row_kind: str = "core",
    hotspot_values: Optional[List[Any]] = None,
) -> Any:
    null_prob = 0.15
    if stress_mode == "null_heavy":
        null_prob = 0.45
    elif stress_mode == "groupby_heavy":
        null_prob = 0.22

    if row_kind == "adversarial":
        null_prob += 0.12
    elif row_kind == "noise":
        null_prob -= 0.03

    if col.nullable and random.random() < max(0.0, min(0.7, null_prob)):
        return None

    if hotspot_values:
        hot_prob = 0.0
        if row_kind == "adversarial":
            hot_prob = 0.85
        elif row_kind == "core":
            hot_prob = 0.55
        elif row_kind == "noise":
            hot_prob = 0.25
        if random.random() < hot_prob:
            return random.choice(hotspot_values)

    if pool:
        reuse_prob = 0.55
        if stress_mode in ("groupby_heavy", "join_heavy", "duplicate_column_heavy"):
            reuse_prob = 0.8
        elif stress_mode == "null_heavy":
            reuse_prob = 0.7
        if row_kind == "adversarial":
            reuse_prob = max(reuse_prob, 0.9)
        if random.random() < reuse_prob:
            return random.choice(pool)

    variant = "normal"
    if row_kind == "noise":
        variant = "wide"
    elif row_kind == "adversarial":
        variant = "stress"
    return _random_scalar_value(col, variant=variant)


def _random_scalar_value(col: Column, variant: str = "normal") -> Any:
    if col.col_type == ColType.INT:
        if variant == "wide":
            return random.randint(-10, 120)
        if variant == "stress":
            return random.choice([0, 1, 2, 42, 50, 77, 99, 100])
        return random.randint(1, 100)

    if col.col_type == ColType.FLOAT:
        if variant == "wide":
            return round(random.uniform(-10.0, 120.0), 1)
        if variant == "stress":
            return random.choice([0.0, 0.1, 1.0, 28.6, 50.0, 77.7, 87.5, 99.9])
        return round(random.uniform(1.0, 100.0), 1)

    if col.col_type == ColType.VARCHAR:
        if variant == "stress":
            return random.choice(["", "a", "dup", "edge", "zzzz"])
        length = random.randint(3, 8)
        return "".join(random.choices(string.ascii_lowercase, k=length))

    return random.randint(1, 100)


def _build_value_pool(
    col: Column,
    stress_mode: str,
    profile: Optional[dict],
) -> List[Any]:
    pool = []
    hot_values = _hot_values_for_column(col, profile)
    if hot_values:
        pool.extend(hot_values[:2])

    pool_size = 2
    if stress_mode in ("groupby_heavy", "join_heavy", "duplicate_column_heavy"):
        pool_size = 4
    elif stress_mode == "null_heavy":
        pool_size = 3

    while len(pool) < pool_size:
        pool.append(_random_scalar_value(col))
    return pool[:pool_size]


def _hot_values_for_column(col: Column, profile: Optional[dict]) -> List[Any]:
    if not profile or col.name not in profile["hot_short_names"]:
        return []

    values = []
    for threshold in profile["thresholds"].get(col.name, []):
        if col.col_type == ColType.INT:
            values.extend([int(threshold) - 1, int(threshold), int(threshold) + 1])
        elif col.col_type == ColType.FLOAT:
            base = float(threshold)
            values.extend([round(base - 0.1, 1), round(base, 1), round(base + 0.1, 1)])

    if col.col_type == ColType.INT:
        values.extend([0, 1, 2, 50, 99, 100])
    elif col.col_type == ColType.FLOAT:
        values.extend([0.0, 0.1, 1.0, 28.6, 77.7, 87.5, 99.9])
    else:
        values.extend(["", "dup", "edge"])

    dedup = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            dedup.append(value)
    return dedup


def _edge_values_for_column(col: Column, profile: Optional[dict]) -> List[Any]:
    hot_values = _hot_values_for_column(col, profile)
    if hot_values:
        return hot_values

    if col.col_type == ColType.INT:
        return [0, 1, -1, 50, 99, 100]
    if col.col_type == ColType.FLOAT:
        return [0.0, 0.1, -1.0, 28.6, 77.7, 99.9]
    return ["", "a", "dup", "edge", "zzzz"]


def _is_fk_col(col_name: str, table: TableSchema) -> bool:
    return any(fk.src_col == col_name for fk in table.fks)


def _get_fk_ref_table(col_name: str, table: TableSchema) -> Optional[str]:
    for fk in table.fks:
        if fk.src_col == col_name:
            return fk.ref_table
    return None


def _generate_with_z3(
    schema: Schema,
    ir: QueryNode,
    rows_per_table: int,
    timeout_sec: int,
    stress_mode: str = "balanced",
    profile: Optional[dict] = None,
) -> Optional[TableData]:
    import z3

    alias_map = _extract_alias_map(ir)
    constraints = _extract_z3_constraints(ir)
    if not constraints or not alias_map:
        return None

    data: TableData = {}
    solver = z3.Solver()
    solver.set("timeout", timeout_sec * 1000)

    z3_vars: Dict[str, Dict[str, Any]] = {}
    table_primary_alias: Dict[str, str] = {}

    for alias, table_name in alias_map.items():
        table = schema.get_table(table_name)
        if table is None:
            continue
        z3_vars[alias] = {}
        table_primary_alias.setdefault(table_name, alias)
        for col in table.non_pk_columns():
            if col.col_type in (ColType.INT, ColType.FLOAT):
                var = z3.Real(f"{alias}_{col.name}_0")
                z3_vars[alias][col.name] = var
                solver.add(var >= 1)
                solver.add(var <= 100)

    for cond in constraints:
        z3_cond = _condition_to_z3(cond, z3_vars, alias_map)
        if z3_cond is not None:
            solver.add(z3_cond)

    if solver.check() != z3.sat:
        return None

    model = solver.model()
    from generator.schema_gen import generate_drop_sqls

    ordered_names = list(reversed(generate_drop_sqls(schema)))
    for table_name in ordered_names:
        table = schema.get_table(table_name)
        first_row_override = None

        primary_alias = table_primary_alias.get(table_name)
        if primary_alias:
            first_row_override = {"id": 1}
            for col in table.columns:
                if col.is_pk or col.name not in z3_vars.get(primary_alias, {}):
                    continue
                z3_val = model.eval(z3_vars[primary_alias][col.name])
                first_row_override[col.name] = _z3_val_to_python(z3_val, col.col_type)

        rows = _generate_table_rows(
            table,
            rows_per_table,
            existing_data=data,
            stress_mode=stress_mode,
            profile=profile,
            first_row_override=first_row_override,
        )
        data[table_name] = rows

    return data


def _condition_to_z3(cond: Condition, z3_vars: dict, alias_map: dict):
    try:
        import z3
    except ImportError:
        return None

    if isinstance(cond, Compare):
        left_var = _resolve_z3_var(cond.field, z3_vars, alias_map)
        if left_var is None:
            return None

        if isinstance(cond.value, str) and "." in cond.value:
            right_val = _resolve_z3_var(cond.value, z3_vars, alias_map)
            if right_val is None:
                return None
        elif isinstance(cond.value, (int, float)):
            right_val = float(cond.value)
        else:
            return None

        if cond.op == CmpOp.EQ:
            return left_var == right_val
        if cond.op == CmpOp.NEQ:
            return left_var != right_val
        if cond.op == CmpOp.GT:
            return left_var > right_val
        if cond.op == CmpOp.GTE:
            return left_var >= right_val
        if cond.op == CmpOp.LT:
            return left_var < right_val
        if cond.op == CmpOp.LTE:
            return left_var <= right_val

    elif isinstance(cond, And):
        left = _condition_to_z3(cond.left, z3_vars, alias_map)
        right = _condition_to_z3(cond.right, z3_vars, alias_map)
        if left is not None and right is not None:
            return z3.And(left, right)
        return left or right

    elif isinstance(cond, Or):
        left = _condition_to_z3(cond.left, z3_vars, alias_map)
        right = _condition_to_z3(cond.right, z3_vars, alias_map)
        if left is not None and right is not None:
            return z3.Or(left, right)
        return left or right

    elif isinstance(cond, Not):
        child = _condition_to_z3(cond.child, z3_vars, alias_map)
        if child is not None:
            return z3.Not(child)

    return None


def _resolve_z3_var(field_name: str, z3_vars: dict, alias_map: dict):
    if "." in field_name:
        alias, col_name = field_name.split(".", 1)
        if alias in alias_map and alias in z3_vars and col_name in z3_vars[alias]:
            return z3_vars[alias][col_name]
    else:
        matches = []
        for cols in z3_vars.values():
            if field_name in cols:
                matches.append(cols[field_name])
        if len(matches) == 1:
            return matches[0]
    return None


def _z3_val_to_python(z3_val, col_type: ColType) -> Any:
    try:
        import z3

        if z3.is_rational_value(z3_val):
            value = z3_val.numerator_as_long() / z3_val.denominator_as_long()
        elif z3.is_int_value(z3_val):
            value = z3_val.as_long()
        else:
            value = float(str(z3_val))

        if col_type == ColType.INT:
            return int(max(-10, min(120, value)))
        if col_type == ColType.FLOAT:
            return round(float(max(-10.0, min(120.0, value))), 1)
        return value
    except Exception:
        return random.randint(1, 100)


def _extract_z3_constraints(ir: QueryNode) -> List[Condition]:
    result = []
    if isinstance(ir, Filter):
        result.append(ir.condition)
        result.extend(_extract_z3_constraints(ir.child))
    elif isinstance(ir, Having):
        result.extend(_extract_z3_constraints(ir.child))
    elif isinstance(ir, (Project, GroupBy)):
        result.extend(_extract_z3_constraints(ir.child))
    elif isinstance(ir, Join):
        result.append(ir.on)
        result.extend(_extract_z3_constraints(ir.left))
        result.extend(_extract_z3_constraints(ir.right))
    return result


def _extract_alias_map(ir: QueryNode) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if isinstance(ir, Scan):
        result[ir.alias] = ir.table
    elif isinstance(ir, Join):
        result.update(_extract_alias_map(ir.left))
        result.update(_extract_alias_map(ir.right))
    elif hasattr(ir, "child"):
        result.update(_extract_alias_map(ir.child))
    return result


def _extract_generation_profile(ir: QueryNode) -> dict:
    alias_map = _extract_alias_map(ir)
    profile = {
        "left_join_right_tables": set(),
        "hot_short_names": set(),
        "nullable_short_names": set(),
        "thresholds": {},
    }

    def collect_aliases(node) -> set:
        if isinstance(node, Scan):
            return {node.alias}
        if isinstance(node, Join):
            return collect_aliases(node.left) | collect_aliases(node.right)
        if hasattr(node, "child"):
            return collect_aliases(node.child)
        return set()

    def mark_field(field_name: str):
        if isinstance(field_name, str) and "." in field_name:
            profile["hot_short_names"].add(field_name.split(".", 1)[1])

    def visit_condition(cond):
        if isinstance(cond, Compare):
            mark_field(cond.field)
            if cond.value is None:
                if "." in cond.field:
                    profile["nullable_short_names"].add(cond.field.split(".", 1)[1])
            elif isinstance(cond.value, (int, float)) and "." in cond.field:
                short_name = cond.field.split(".", 1)[1]
                profile["thresholds"].setdefault(short_name, []).append(cond.value)
            elif isinstance(cond.value, str) and "." in cond.value:
                mark_field(cond.value)
            return
        if isinstance(cond, (And, Or)):
            visit_condition(cond.left)
            visit_condition(cond.right)
            return
        if isinstance(cond, Not):
            visit_condition(cond.child)

    def visit(node):
        if isinstance(node, Join):
            visit_condition(node.on)
            if node.join_type == JoinType.LEFT:
                right_aliases = collect_aliases(node.right)
                for alias in right_aliases:
                    table_name = alias_map.get(alias)
                    if table_name:
                        profile["left_join_right_tables"].add(table_name)
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, Filter):
            visit_condition(node.condition)
            visit(node.child)
            return
        if isinstance(node, GroupBy):
            for field in node.fields:
                mark_field(field)
            for agg in node.aggregates:
                if agg.field != "*":
                    mark_field(agg.field)
            visit(node.child)
            return
        if isinstance(node, Having):
            visit_condition(node.condition)
            visit(node.child)
            return
        if isinstance(node, Project):
            for field in node.fields:
                mark_field(field)
            visit(node.child)

    visit(ir)
    return profile


def _insert_all(schema: Schema, data: TableData) -> None:
    from db.connector import insert_rows
    from generator.schema_gen import generate_drop_sqls

    ordered_names = list(reversed(generate_drop_sqls(schema)))
    for table_name in ordered_names:
        if table_name not in data or not data[table_name]:
            continue
        table = schema.get_table(table_name)
        columns = [col.name for col in table.columns]
        rows = [tuple(row[col] for col in columns) for row in data[table_name]]
        insert_rows(table_name, columns, rows)
