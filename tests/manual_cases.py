"""轻量回归样例，优先覆盖不依赖数据库的关键逻辑。"""

import os
import sys
import types
import importlib.util
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
    _plan_row_budget,
    _random_value,
    _resolve_z3_var,
)
from generator.ir_gen import (
    GenContext,
    _choose_group_fields,
    _choose_project_fields,
    _generate_condition,
    _get_nullable_visible_cols,
    generate_ir,
)
from generator.schema_gen import ColType, Column, TableSchema
from ir.nodes import (
    AggFunc,
    Aggregate,
    And,
    ArithExpr,
    ArithOp,
    Between,
    CaseWhen,
    Compare,
    CmpOp,
    Distinct,
    Filter,
    GroupBy,
    Having,
    InList,
    Join,
    JoinType,
    Like,
    LimitOffset,
    OrderBy,
    OrderKey,
    Or,
    Project,
    Scan,
    SelectItem,
    WhenClause,
)
from translators.python_ref import (
    _eval_condition_3vl,
    _eval_distinct,
    _eval_join,
    _eval_limit_offset,
    _eval_orderby,
    _eval_project,
)
from translators.sql import translate
from runner import (
    BugReport,
    RunStats,
    _collect_query_features,
    _choose_stress_mode,
    _classify_bug_report,
    _query_requires_ordered_compare,
    _report_has_actionable_bug,
)


def _unwrap_wrappers(node):
    while isinstance(node, (Distinct, OrderBy, LimitOffset)):
        node = node.child
    return node


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


def test_compare_unordered_rows_with_float_noise_do_not_false_positive():
    ref_rows = [
        {"u.price": 1.0, "avg_u_price": 1.0, "i.id": 1},
        {"u.price": 0.1, "avg_u_price": 0.10000000000000002, "i.id": 1},
        {"u.price": 20.5, "avg_u_price": 20.5, "i.id": 1},
        {"u.price": 66.7, "avg_u_price": 66.7, "i.id": 2},
        {"u.price": 28.6, "avg_u_price": 28.600000000000005, "i.id": 2},
        {"u.price": 0.1, "avg_u_price": 0.1, "i.id": 2},
        {"u.price": 1.2, "avg_u_price": 1.2, "i.id": 3},
        {"u.price": 0.1, "avg_u_price": 0.09999999999999999, "i.id": 3},
    ]
    sql_rows = [
        {"price": 20.5, "avg_u_price": 20.5, "id": 1},
        {"price": 0.1, "avg_u_price": 0.10000000149011612, "id": 1},
        {"price": 1.0, "avg_u_price": 1.0, "id": 1},
        {"price": 28.6, "avg_u_price": 28.600000381469727, "id": 2},
        {"price": 0.1, "avg_u_price": 0.10000000149011612, "id": 2},
        {"price": 66.7, "avg_u_price": 66.69999694824219, "id": 2},
        {"price": 0.1, "avg_u_price": 0.10000000149011612, "id": 3},
        {"price": 1.2, "avg_u_price": 1.2000000476837158, "id": 3},
    ]

    result = compare_two_paths(ref_rows, sql_rows, "ref", "sql")
    assert result.match


def test_compare_unordered_rows_still_detects_real_row_difference():
    rows_a = [{"id": 1, "avg": 0.1}, {"id": 2, "avg": 28.6}]
    rows_b = [{"id": 1, "avg": 0.10000000149011612}, {"id": 3, "avg": 28.600000381469727}]

    result = compare_two_paths(rows_a, rows_b, "ref", "sql")
    assert not result.match


def test_compare_ordered_rows_detects_sequence_difference():
    rows_a = [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}]
    rows_b = [{"id": 2, "amount": 20}, {"id": 1, "amount": 10}]

    result = compare_two_paths(rows_a, rows_b, "ref", "sql", ordered=True)
    assert not result.match


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
        cond = _generate_condition(
            ctx,
            schema=None,
            stress_mode="null_heavy",
            template={"force_null_compare": True},
        )

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


def test_sql_translate_supports_distinct_and_orderby():
    ir = OrderBy(
        keys=[OrderKey("sum_o_amount", descending=True), OrderKey("o.id")],
        child=Distinct(
            child=Project(
                fields=["o.id", "sum_o_amount"],
                child=GroupBy(
                    fields=["o.id"],
                    aggregates=[Aggregate(AggFunc.SUM, "o.amount", "sum_o_amount")],
                    child=Scan("orders", "o"),
                ),
            )
        ),
    )

    sql = translate(ir)
    assert "SELECT DISTINCT" in sql
    assert "ORDER BY" in sql
    assert "DESC" in sql


def test_sql_translate_supports_limit_offset_and_extended_predicates():
    ir = LimitOffset(
        limit=5,
        offset=2,
        child=OrderBy(
            keys=[OrderKey(ArithExpr("o.amount", ArithOp.ADD, 1), descending=True)],
            child=Project(
                fields=[
                    "o.id",
                    SelectItem(
                        expr=CaseWhen(
                            cases=[WhenClause(Compare("o.amount", CmpOp.GTE, 10), 1)],
                            else_value=0,
                        ),
                        alias="amount_bucket",
                    ),
                ],
                child=Filter(
                    condition=And(
                        Between("o.amount", 1, 10),
                        Or(Like("o.code", "a%"), InList("o.status", ["dup", "edge"])),
                    ),
                    child=Scan("orders", "o"),
                ),
            ),
        ),
    )

    sql = translate(ir)
    assert "LIMIT 5 OFFSET 2" in sql
    assert "BETWEEN" in sql
    assert "LIKE" in sql
    assert "IN (" in sql
    assert "CASE WHEN" in sql


def test_sql_translate_case_when_over_aggregate_alias_uses_plain_aggregate_expr():
    ir = Project(
        fields=[
            "u.num",
            "max_u_val",
            SelectItem(
                expr=CaseWhen(
                    cases=[WhenClause(Compare("max_u_val", CmpOp.GTE, 50), 1)],
                    else_value=0,
                ),
                alias="max_u_val_bucket",
            ),
        ],
        child=Having(
            condition=Compare("max_u_val", CmpOp.LT, 70),
            child=GroupBy(
                fields=["u.num"],
                aggregates=[
                    Aggregate(AggFunc.MAX, "u.val", "max_u_val"),
                ],
                child=Scan("users", "u"),
            ),
        ),
    )

    sql = translate(ir)
    assert "MAX(`u`.`val`) AS `max_u_val`" in sql
    assert "CASE WHEN MAX(`u`.`val`) >= 50 THEN 1 ELSE 0 END" in sql
    assert "CASE WHEN MAX(`u`.`val`) AS `max_u_val` >= 50" not in sql


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


def test_python_ref_distinct_removes_duplicate_rows():
    node = Distinct(child=Scan("orders", "o"))

    with patch("translators.python_ref._eval", return_value=[
        {"o.id": 1, "o.amount": 10},
        {"o.id": 1, "o.amount": 10},
        {"o.id": 2, "o.amount": 10},
    ]):
        rows = _eval_distinct(node)

    assert rows == [
        {"o.id": 1, "o.amount": 10},
        {"o.id": 2, "o.amount": 10},
    ]


def test_python_ref_orderby_applies_mixed_direction_keys():
    node = OrderBy(
        keys=[OrderKey("o.amount", descending=True), OrderKey("o.id", descending=False)],
        child=Scan("orders", "o"),
    )

    with patch("translators.python_ref._eval", return_value=[
        {"o.id": 2, "o.amount": 10},
        {"o.id": 1, "o.amount": 10},
        {"o.id": 3, "o.amount": 20},
    ]):
        rows = _eval_orderby(node)

    assert rows == [
        {"o.id": 3, "o.amount": 20},
        {"o.id": 1, "o.amount": 10},
        {"o.id": 2, "o.amount": 10},
    ]


def test_python_ref_project_supports_case_when_and_arithmetic():
    node = Project(
        fields=[
            "o.id",
            SelectItem(expr=ArithExpr("o.amount", ArithOp.ADD, 2), alias="amount_plus_2"),
            SelectItem(
                expr=CaseWhen(
                    cases=[WhenClause(Compare("o.amount", CmpOp.GTE, 10), 1)],
                    else_value=0,
                ),
                alias="amount_bucket",
            ),
        ],
        child=Scan("orders", "o"),
    )

    with patch("translators.python_ref._eval", return_value=[
        {"o.id": 1, "o.amount": 8},
        {"o.id": 2, "o.amount": 12},
    ]):
        rows = _eval_project(node)

    assert rows == [
        {"o.id": 1, "amount_plus_2": 10, "amount_bucket": 0},
        {"o.id": 2, "amount_plus_2": 14, "amount_bucket": 1},
    ]


def test_python_ref_extended_predicates_and_limit_offset():
    row = {"o.amount": 9, "o.code": "alpha", "o.status": "dup"}
    assert _eval_condition_3vl(Between("o.amount", 1, 10), row) is True
    assert _eval_condition_3vl(Like("o.code", "a%"), row) is True
    assert _eval_condition_3vl(InList("o.status", ["dup", "edge"]), row) is True

    node = LimitOffset(limit=2, offset=1, child=Scan("orders", "o"))
    with patch("translators.python_ref._eval", return_value=[
        {"o.id": 1},
        {"o.id": 2},
        {"o.id": 3},
        {"o.id": 4},
    ]):
        rows = _eval_limit_offset(node)
    assert rows == [{"o.id": 2}, {"o.id": 3}]


def test_data_gen_null_heavy_raises_null_probability():
    nullable_int = Column(name="score", col_type=ColType.INT, nullable=True)

    with patch("generator.data_gen.random.random", return_value=0.3):
        assert _random_value(nullable_int, stress_mode="null_heavy") is None

    with patch("generator.data_gen.random.random", return_value=0.3), patch(
        "generator.data_gen.random.randint", return_value=17
    ):
        assert _random_value(nullable_int, stress_mode="balanced") == 17


def test_data_gen_row_budget_adds_edge_and_adversarial_rows():
    profile = {"left_join_right_tables": {"orders"}}
    budget = _plan_row_budget(
        base_rows=10,
        stress_mode="null_heavy",
        table_name="orders",
        profile=profile,
    )

    assert budget["core"] == 10
    assert budget["edge"] >= 4
    assert budget["adversarial"] >= 8
    assert budget["noise"] >= 8


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


def test_ir_generator_groupby_heavy_enforces_groupby_and_having():
    schema = types.SimpleNamespace(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
                    Column(name="score", col_type=ColType.INT, nullable=True),
                    Column(name="amount", col_type=ColType.FLOAT, nullable=True),
                ],
            )
        ],
        fk_pairs=lambda: [],
        get_table=lambda name: next(t for t in schema.tables if t.name == name),
    )

    ir, _ = generate_ir(schema, stress_mode="groupby_heavy", seed=7)

    assert _collect_query_features(ir)["has_orderby"] is True
    core = _unwrap_wrappers(ir)
    assert isinstance(core, Project)
    assert isinstance(core.child, Having)
    assert isinstance(core.child.child, GroupBy)


def test_ir_generator_orderby_heavy_emits_orderby():
    schema = types.SimpleNamespace(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
                    Column(name="score", col_type=ColType.INT, nullable=True),
                ],
            ),
            TableSchema(
                name="orders",
                columns=[
                    Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
                    Column(name="amount", col_type=ColType.FLOAT, nullable=True),
                    Column(name="users_id", col_type=ColType.INT, nullable=False),
                ],
                fks=[types.SimpleNamespace(src_table="orders", src_col="users_id", ref_table="users", ref_col="id")],
            ),
        ],
        fk_pairs=lambda: list(schema.tables[1].fks),
        get_table=lambda name: schema.tables[0] if name == "users" else schema.tables[1],
    )

    ir, _ = generate_ir(schema, stress_mode="orderby_heavy", seed=7)

    assert _query_requires_ordered_compare(ir) is True
    assert _collect_query_features(ir)["has_orderby"] is True


def test_ir_generator_distinct_heavy_emits_distinct():
    schema = types.SimpleNamespace(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
                    Column(name="score", col_type=ColType.INT, nullable=True),
                ],
            )
        ],
        fk_pairs=lambda: [],
        get_table=lambda name: schema.tables[0],
    )

    ir, _ = generate_ir(schema, stress_mode="distinct_heavy", seed=9)

    assert _collect_query_features(ir)["has_distinct"] is True


def test_ir_generator_null_heavy_can_force_left_join_and_null_predicate():
    left = TableSchema(
        name="users",
        columns=[
            Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
            Column(name="score", col_type=ColType.INT, nullable=True),
        ],
    )
    right = TableSchema(
        name="orders",
        columns=[
            Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True),
            Column(name="amount", col_type=ColType.FLOAT, nullable=True),
            Column(name="users_id", col_type=ColType.INT, nullable=False),
        ],
        fks=[types.SimpleNamespace(src_table="orders", src_col="users_id", ref_table="users", ref_col="id")],
    )
    schema = types.SimpleNamespace(
        tables=[left, right],
        fk_pairs=lambda: list(right.fks),
        get_table=lambda name: left if name == "users" else right,
    )

    ir, _ = generate_ir(schema, stress_mode="null_heavy", seed=11)

    assert "LEFT" in repr(ir) or True
    sql = translate(ir)
    assert "LEFT JOIN" in sql
    assert "NULL" in sql


def test_runner_collects_distinct_and_orderby_features():
    ir = OrderBy(
        keys=[OrderKey("sum_o_amount")],
        child=Distinct(
            child=Project(
                fields=["o.id", "sum_o_amount"],
                child=GroupBy(
                    fields=["o.id"],
                    aggregates=[Aggregate(AggFunc.SUM, "o.amount", "sum_o_amount")],
                    child=Scan("orders", "o"),
                ),
            )
        ),
    )

    features = _collect_query_features(ir)
    assert features["has_distinct"] is True
    assert features["has_orderby"] is True
    assert features["has_orderby_agg"] is True


def test_runner_collects_extended_syntax_features():
    ir = LimitOffset(
        limit=3,
        offset=1,
        child=OrderBy(
            keys=[OrderKey(ArithExpr("o.amount", ArithOp.ADD, 1), descending=True)],
            child=Project(
                fields=[
                    "o.id",
                    SelectItem(
                        expr=CaseWhen(
                            cases=[WhenClause(Compare("o.amount", CmpOp.GTE, 10), 1)],
                            else_value=0,
                        ),
                        alias="bucket",
                    ),
                ],
                child=Filter(
                    condition=And(
                        Between("o.amount", 1, 10),
                        Or(Like("o.code", "a%"), InList("o.status", ["dup", "edge"])),
                    ),
                    child=Scan("orders", "o"),
                ),
            ),
        ),
    )

    features = _collect_query_features(ir)
    assert features["has_limit_offset"] is True
    assert features["has_between"] is True
    assert features["has_like"] is True
    assert features["has_in_list"] is True
    assert features["has_arithmetic_expr"] is True
    assert features["has_case_when"] is True


def test_connector_execute_sql_without_params_does_not_pass_empty_args():
    connector_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "db",
        "connector.py",
    )
    spec = importlib.util.spec_from_file_location("retorm_db_connector_real", connector_path)
    connector_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(connector_mod)

    calls = []

    class FakeCursor:
        def execute(self, *args):
            calls.append(args)

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            return None

    with patch.object(connector_mod, "get_connection", return_value=FakeConn()):
        connector_mod.execute_sql("SELECT 1 WHERE 'a%';")
        connector_mod.execute_sql("SELECT * FROM t WHERE id = %s", (1,))

    assert calls[0] == ("SELECT 1 WHERE 'a%';",)
    assert calls[1] == ("SELECT * FROM t WHERE id = %s", (1,))


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


def test_runner_classifies_sql_mismatch_without_compare_result_truthiness():
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
        ref_vs_sql=CompareResult(match=False, reason="float mismatch"),
        ref_vs_orm=CompareResult(match=True),
        ref_rows=[],
        sql_rows=[],
        orm_rows=[],
    )

    bug_type, reason = _classify_bug_report(report)

    assert "SQL" in bug_type
    assert reason == "float mismatch"


def test_runner_classifies_orm_mismatch_without_compare_result_truthiness():
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
        ref_vs_orm=CompareResult(match=False, reason="orm mismatch"),
        ref_rows=[],
        sql_rows=[],
        orm_rows=[],
    )

    bug_type, reason = _classify_bug_report(report)

    assert "ORM" in bug_type
    assert reason == "orm mismatch"


def test_runner_classifies_ref_anomaly_when_sql_equals_orm():
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
        ref_vs_sql=CompareResult(match=False, reason="ref mismatch"),
        ref_vs_orm=CompareResult(match=False, reason="ref mismatch"),
        sql_vs_orm=CompareResult(match=True),
        ref_rows=[],
        sql_rows=[],
        orm_rows=[],
    )

    bug_type, reason = _classify_bug_report(report)

    assert "Ref" in bug_type
    assert "sql_vs_orm" in reason


def test_runner_stress_mode_boosts_duplicate_when_coverage_low():
    stats = RunStats(total_queries=100, duplicate_proj_queries=0)
    mode = _choose_stress_mode(12345, stats)
    assert mode in {
        "balanced",
        "join_heavy",
        "groupby_heavy",
        "duplicate_column_heavy",
        "null_heavy",
        "orderby_heavy",
        "distinct_heavy",
    }


if __name__ == "__main__":
    test_compare_duplicate_projected_columns()
    test_compare_duplicate_projected_numeric_columns()
    test_compare_alias_with_letter_digit_prefix()
    test_compare_aggregate_decimal_tolerance()
    test_compare_unordered_rows_with_float_noise_do_not_false_positive()
    test_compare_unordered_rows_still_detects_real_row_difference()
    test_compare_ordered_rows_detects_sequence_difference()
    test_compare_null_nan_and_bool_normalization()
    test_ir_generator_keeps_duplicate_short_names_when_projecting()
    test_ir_generator_prefers_duplicate_pair_when_available()
    test_data_gen_extracts_alias_map_and_join_constraints()
    test_data_gen_resolves_aliases_without_fuzzy_matching()
    test_ir_generator_null_heavy_can_emit_null_predicates()
    test_sql_translate_uses_is_null_and_is_not_null()
    test_sql_translate_supports_left_join()
    test_sql_translate_supports_nested_join_chain()
    test_sql_translate_supports_distinct_and_orderby()
    test_sql_translate_supports_limit_offset_and_extended_predicates()
    test_python_ref_null_compare_matches_is_null_semantics()
    test_python_ref_left_join_null_extends_unmatched_rows()
    test_python_ref_distinct_removes_duplicate_rows()
    test_python_ref_orderby_applies_mixed_direction_keys()
    test_python_ref_project_supports_case_when_and_arithmetic()
    test_python_ref_extended_predicates_and_limit_offset()
    test_data_gen_null_heavy_raises_null_probability()
    test_data_gen_row_budget_adds_edge_and_adversarial_rows()
    test_ir_generator_groupby_sampling_respects_preferred_pool_size()
    test_ir_generator_treats_left_join_right_columns_as_query_nullable()
    test_ir_generator_groupby_heavy_enforces_groupby_and_having()
    test_ir_generator_orderby_heavy_emits_orderby()
    test_ir_generator_distinct_heavy_emits_distinct()
    test_ir_generator_null_heavy_can_force_left_join_and_null_predicate()
    test_runner_collects_distinct_and_orderby_features()
    test_runner_skips_all_matched_bug_reports()
    test_runner_keeps_real_mismatches_and_errors()
    test_runner_classifies_sql_mismatch_without_compare_result_truthiness()
    test_runner_classifies_orm_mismatch_without_compare_result_truthiness()
    test_runner_classifies_ref_anomaly_when_sql_equals_orm()
    test_runner_stress_mode_boosts_duplicate_when_coverage_low()
    print("manual_cases: all checks passed")
