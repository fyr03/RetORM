"""轻量回归样例，优先覆盖不依赖数据库的关键逻辑。"""

import os
import sys
import types
from decimal import Decimal
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "db.connector" not in sys.modules:
    connector_stub = types.ModuleType("db.connector")
    connector_stub.execute_sql = lambda *args, **kwargs: []
    connector_stub.get_engine = lambda *args, **kwargs: None
    connector_stub.init_database = lambda *args, **kwargs: None
    connector_stub.create_tables = lambda *args, **kwargs: None
    connector_stub.drop_tables = lambda *args, **kwargs: None
    connector_stub.dispose_engine = lambda *args, **kwargs: None
    sys.modules["db.connector"] = connector_stub

from comparator.compare import CompareResult, compare_two_paths, normalize
from generator.data_gen import (
    _extract_alias_map,
    _extract_z3_constraints,
    _random_value,
    _resolve_z3_var,
)
from generator.ir_gen import (
    GenContext,
    _choose_group_fields,
    _choose_project_fields,
    _generate_condition,
    _get_nullable_visible_cols,
)
from generator.schema_gen import ColType, Column, TableSchema
from ir.nodes import Compare, CmpOp, Filter, Join, JoinType, Project, Scan
from translators.python_ref import _eval_condition_3vl, _eval_join
from translators.sql import translate
from runner import BugReport, _report_has_actionable_bug


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


def test_ir_generator_null_heavy_can_emit_null_predicates():
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

    with patch("generator.ir_gen.random.random", side_effect=[0.0, 0.0]), patch(
        "generator.ir_gen.random.choice",
        side_effect=["i.nullable_score", CmpOp.EQ],
    ):
        cond = _generate_condition(ctx, schema=None, stress_mode="null_heavy")

    assert isinstance(cond, Compare)
    assert cond.field == "i.nullable_score"
    assert cond.value is None


def test_sql_translate_uses_is_null_and_is_not_null():
    ir_is_null = Project(
        fields=["u.id"],
        child=Filter(
            condition=Compare("u.score", CmpOp.EQ, None),
            child=Scan("users", "u"),
        ),
    )
    ir_is_not_null = Project(
        fields=["u.id"],
        child=Filter(
            condition=Compare("u.score", CmpOp.NEQ, None),
            child=Scan("users", "u"),
        ),
    )

    assert "IS NULL" in translate(ir_is_null)
    assert "IS NOT NULL" in translate(ir_is_not_null)


def test_sql_translate_supports_left_join():
    ir = Project(
        fields=["o.id", "u.score"],
        child=Join(
            left=Scan("orders", "o"),
            right=Scan("users", "u"),
            on=Compare("o.user_id", CmpOp.EQ, "u.id"),
            join_type=JoinType.LEFT,
        ),
    )

    assert "LEFT JOIN" in translate(ir)


def test_sql_translate_supports_nested_join_chain():
    ir = Project(
        fields=["o.id", "u.score", "p.amount"],
        child=Join(
            left=Join(
                left=Scan("orders", "o"),
                right=Scan("users", "u"),
                on=Compare("o.user_id", CmpOp.EQ, "u.id"),
                join_type=JoinType.LEFT,
            ),
            right=Scan("payments", "p"),
            on=Compare("p.order_id", CmpOp.EQ, "o.id"),
            join_type=JoinType.INNER,
        ),
    )

    sql = translate(ir)
    assert sql.count("JOIN") == 2
    assert "LEFT JOIN" in sql
    assert "INNER JOIN" in sql


def test_python_ref_null_compare_matches_is_null_semantics():
    row = {"u.score": None, "u.id": 1}
    not_null_row = {"u.score": 7, "u.id": 2}

    assert _eval_condition_3vl(Compare("u.score", CmpOp.EQ, None), row) is True
    assert _eval_condition_3vl(Compare("u.score", CmpOp.EQ, None), not_null_row) is False
    assert _eval_condition_3vl(Compare("u.score", CmpOp.NEQ, None), row) is False
    assert _eval_condition_3vl(Compare("u.score", CmpOp.NEQ, None), not_null_row) is True


def test_python_ref_left_join_null_extends_unmatched_rows():
    join = Join(
        left=Scan("orders", "o"),
        right=Scan("users", "u"),
        on=Compare("o.user_id", CmpOp.EQ, "u.id"),
        join_type=JoinType.LEFT,
    )

    def _fake_eval(node):
        if isinstance(node, Scan) and node.alias == "o":
            return [
                {"o.id": 1, "o.user_id": 1},
                {"o.id": 2, "o.user_id": 99},
            ]
        if isinstance(node, Scan) and node.alias == "u":
            return [
                {"u.id": 1, "u.score": 7},
            ]
        raise AssertionError(f"unexpected node: {node}")

    with patch("translators.python_ref._eval", side_effect=_fake_eval):
        rows = _eval_join(join)

    assert rows == [
        {"o.id": 1, "o.user_id": 1, "u.id": 1, "u.score": 7},
        {"o.id": 2, "o.user_id": 99, "u.id": None, "u.score": None},
    ]


def test_data_gen_null_heavy_raises_null_probability():
    nullable_int = Column(name="score", col_type=ColType.INT, nullable=True)

    with patch("generator.data_gen.random.random", return_value=0.3):
        assert _random_value(nullable_int, stress_mode="null_heavy") is None

    with patch("generator.data_gen.random.random", return_value=0.3), patch(
        "generator.data_gen.random.randint", return_value=17
    ):
        assert _random_value(nullable_int, stress_mode="balanced") == 17


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


def test_ir_generator_treats_left_join_right_columns_as_query_nullable():
    table = TableSchema(
        name="users",
        columns=[
            Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
            Column(name="score", col_type=ColType.INT, nullable=False),
        ],
    )
    ctx = GenContext(tables={"u": table}, visible_cols=["u.id", "u.score"])
    ctx.query_nullable_cols.add("u.score")

    assert _get_nullable_visible_cols(ctx) == ["u.score"]


def test_runner_skips_all_matched_bug_reports():
    report = BugReport(
        schema_id=0,
        query_id=0,
        schema=None,
        ir=None,
        ir_str="",
        schema_seed=1,
        query_seed=2,
        table_data={},
        rows_per_table=0,
        use_z3=False,
        z3_timeout=0,
        ref_vs_sql=CompareResult(match=True),
        ref_vs_orm=CompareResult(match=True),
        ref_rows=[],
        sql_rows=[],
        orm_rows=[],
    )

    assert _report_has_actionable_bug(report) is False


def test_runner_keeps_real_mismatches_and_errors():
    mismatch_report = BugReport(
        schema_id=0,
        query_id=0,
        schema=None,
        ir=None,
        ir_str="",
        schema_seed=1,
        query_seed=2,
        table_data={},
        rows_per_table=0,
        use_z3=False,
        z3_timeout=0,
        ref_vs_sql=CompareResult(match=False, reason="mismatch"),
        ref_vs_orm=CompareResult(match=True),
        ref_rows=[],
        sql_rows=[],
        orm_rows=[],
    )
    error_report = BugReport(
        schema_id=0,
        query_id=0,
        schema=None,
        ir=None,
        ir_str="",
        schema_seed=1,
        query_seed=2,
        table_data={},
        rows_per_table=0,
        use_z3=False,
        z3_timeout=0,
        ref_vs_sql=None,
        ref_vs_orm=None,
        ref_rows=[],
        sql_rows=[],
        orm_rows=[],
        error="orm unsupported query",
    )

    assert _report_has_actionable_bug(mismatch_report) is True
    assert _report_has_actionable_bug(error_report) is True


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
    test_ir_generator_null_heavy_can_emit_null_predicates()
    test_sql_translate_uses_is_null_and_is_not_null()
    test_sql_translate_supports_left_join()
    test_sql_translate_supports_nested_join_chain()
    test_python_ref_null_compare_matches_is_null_semantics()
    test_python_ref_left_join_null_extends_unmatched_rows()
    test_data_gen_null_heavy_raises_null_probability()
    test_ir_generator_groupby_sampling_respects_preferred_pool_size()
    test_ir_generator_treats_left_join_right_columns_as_query_nullable()
    test_runner_skips_all_matched_bug_reports()
    test_runner_keeps_real_mismatches_and_errors()
    print("manual_cases: all checks passed")
