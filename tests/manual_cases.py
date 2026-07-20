"""轻量回归样例，优先覆盖不依赖数据库的关键逻辑。"""

import os
import sys
from decimal import Decimal
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparator.compare import compare_two_paths, normalize
from generator.data_gen import _extract_alias_map, _extract_z3_constraints, _resolve_z3_var
from generator.ir_gen import (
    GenContext,
    _choose_group_fields,
    _choose_project_fields,
    _generate_condition,
)
from generator.schema_gen import ColType, Column, TableSchema
from ir.nodes import Compare, CmpOp, Filter, Join, Scan


def test_compare_duplicate_projected_columns():
    ref_rows = [{"o.id": 1, "u.id": 2, "o.amount": 50.0}]
    sql_rows = [{"id": 1, "u.id": 2, "amount": 50.0}]
    orm_rows = [{"o_id": 1, "u_id": 2, "o_amount": 50.0}]

    assert normalize(ref_rows) == [{"id#1": 1, "id#2": 2, "amount": 50.0}]
    assert compare_two_paths(ref_rows, sql_rows, "ref", "sql").match
    assert compare_two_paths(ref_rows, orm_rows, "ref", "orm").match


def test_compare_duplicate_projected_numeric_columns():
    ref_rows = [{"c.amount": 10.0, "l.amount": 20.0}]
    sql_rows = [{"amount": 10.0, "l.amount": 20.0}]
    orm_rows = [{"c_amount": 10.0, "l_amount": 20.0}]

    assert normalize(ref_rows) == [{"amount#1": 10.0, "amount#2": 20.0}]
    assert compare_two_paths(ref_rows, sql_rows, "ref", "sql").match
    assert compare_two_paths(ref_rows, orm_rows, "ref", "orm").match


def test_compare_alias_with_letter_digit_prefix():
    row = {"p2_num": 7, "u3_id": 9, "count_all": 11}
    assert normalize([row]) == [{"num": 7, "id": 9, "count_all": 11}]


def test_compare_aggregate_decimal_tolerance():
    ref_rows = [{"avg_amount": 7.3333333333, "sum_amount": 22.0}]
    sql_rows = [{"avg_amount": Decimal("7.3333"), "sum_amount": Decimal("22.0000")}]

    assert compare_two_paths(ref_rows, sql_rows, "ref", "sql").match


def test_compare_null_nan_and_bool_normalization():
    row = {"o.flag": True, "o.score": float("nan"), "o.amount": None}
    assert normalize([row]) == [{"flag": 1, "score": None, "amount": None}]


def test_ir_generator_keeps_duplicate_short_names_when_projecting():
    ctx = GenContext(visible_cols=["o.id", "u.id", "o.amount"])

    with patch("generator.ir_gen.random.random", return_value=1.0), patch(
        "generator.ir_gen.random.randint", return_value=2
    ), patch("generator.ir_gen.random.sample", return_value=["o.id", "u.id"]):
        fields = _choose_project_fields(ctx)

    assert fields == ["o.id", "u.id"]


def test_ir_generator_prefers_duplicate_pair_when_available():
    ctx = GenContext(visible_cols=["o.id", "u.id", "o.amount", "u.amount"])

    with patch("generator.ir_gen.random.random", return_value=0.0), patch(
        "generator.ir_gen.random.choice", return_value=["o.id", "u.id"]
    ), patch("generator.ir_gen.random.randint", return_value=0):
        fields = _choose_project_fields(ctx)

    assert fields == ["o.id", "u.id"]


def test_data_gen_extracts_alias_map_and_join_constraints():
    ir = Filter(
        condition=Compare("c.amount", CmpOp.GT, 10),
        child=Join(
            left=Scan("categories", "c"),
            right=Scan("logs", "l"),
            on=Compare("l.categories_id", CmpOp.EQ, "c.id"),
        ),
    )

    assert _extract_alias_map(ir) == {"c": "categories", "l": "logs"}
    constraints = _extract_z3_constraints(ir)
    assert len(constraints) == 2
    assert constraints[0] == ir.condition
    assert constraints[1] == ir.child.on


def test_data_gen_resolves_aliases_without_fuzzy_matching():
    z3_vars = {
        "c": {"amount": "c_amount_var"},
        "l": {"amount": "l_amount_var"},
    }
    alias_map = {"c": "categories", "l": "logs"}

    assert _resolve_z3_var("c.amount", z3_vars, alias_map) == "c_amount_var"
    assert _resolve_z3_var("l.amount", z3_vars, alias_map) == "l_amount_var"
    assert _resolve_z3_var("amount", z3_vars, alias_map) is None


def test_ir_generator_null_heavy_prefers_nullable_numeric_columns():
    table = TableSchema(
        name="items",
        columns=[
            Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
            Column(name="nullable_score", col_type=ColType.INT, nullable=True),
            Column(name="strict_amount", col_type=ColType.INT, nullable=False),
        ],
    )
    ctx = GenContext(
        visible_cols=["i.id", "i.nullable_score", "i.strict_amount"],
        tables={"i": table},
    )

    with patch("generator.ir_gen.random.random", return_value=0.0), patch(
        "generator.ir_gen.random.choice",
        side_effect=["i.nullable_score", CmpOp.EQ],
    ), patch("generator.ir_gen.random.randint", return_value=42):
        cond = _generate_condition(ctx, schema=None, stress_mode="null_heavy")

    assert isinstance(cond, Compare)
    assert cond.field == "i.nullable_score"
    assert cond.value == 42


def test_ir_generator_groupby_sampling_respects_preferred_pool_size():
    table = TableSchema(
        name="items",
        columns=[
            Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
            Column(name="nullable_score", col_type=ColType.INT, nullable=True),
            Column(name="strict_amount", col_type=ColType.INT, nullable=False),
        ],
    )
    ctx = GenContext(
        visible_cols=["i.id", "i.nullable_score", "i.strict_amount"],
        tables={"i": table},
    )

    def _assert_group_randint(start, end):
        assert (start, end) == (1, 1)
        return 1

    with patch("generator.ir_gen.random.randint", side_effect=_assert_group_randint), patch(
        "generator.ir_gen.random.sample", return_value=["i.nullable_score"]
    ) as sample_mock:
        fields = _choose_group_fields(ctx, "null_heavy", ctx.visible_cols)

    assert fields == ["i.nullable_score"]
    sample_mock.assert_called_once_with(["i.nullable_score"], 1)


if __name__ == "__main__":
    test_compare_duplicate_projected_columns()
    test_compare_duplicate_projected_numeric_columns()
    test_compare_alias_with_letter_digit_prefix()
    test_compare_aggregate_decimal_tolerance()
    test_compare_null_nan_and_bool_normalization()
    test_ir_generator_keeps_duplicate_short_names_when_projecting()
    test_ir_generator_prefers_duplicate_pair_when_available()
    test_data_gen_extracts_alias_map_and_join_constraints()
    test_data_gen_resolves_aliases_without_fuzzy_matching()
    test_ir_generator_null_heavy_prefers_nullable_numeric_columns()
    test_ir_generator_groupby_sampling_respects_preferred_pool_size()
    print("manual_cases: all checks passed")
