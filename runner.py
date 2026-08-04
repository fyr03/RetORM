"""
runner.py

RetORM main entrypoint.

log:
  logs_detail/YYMMDD_HHMMSS.log  detail log, same with --verbose terminal output
  logs/YYMMDD_HHMMSS.log         running log, output summary per 10 sec
  logs_bug/YYMMDD_HHMMSS.log     bug detail log(IR/SQL/ORM API/program code)

bug output:
  bugs/YYMMDD_HHMMSS/bug_N.py    reproduce script for bug N, can be run directly with python

usage:
  python runner.py
  python runner.py --schemas 3 --queries 10 --seed 42 --no-z3 --verbose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import math
import random
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
from generator.schema_gen import (
    Schema, generate_schema, generate_create_sqls,
    generate_drop_sqls, print_schema,
)
from generator.ir_gen   import generate_ir
from generator.data_gen import generate_and_insert
from translators.python_ref     import execute as ref_execute
from translators.sql            import execute as sql_execute, translate as sql_translate
from translators.sqlalchemy_true_orm import (
    execute as true_orm_execute,
    get_true_orm_coverage_snapshot,
    reset_model_cache,
    reset_true_orm_coverage,
    supports_true_orm,
    UnsupportedTrueORM,
)
from comparator.compare         import CompareResult, compare_all, compare_two_paths, print_report
from db.connector import (
    init_database, create_tables, drop_tables,
    execute_sql, dispose_engine,
)
from ir.nodes import (
    ArithExpr,
    ArithOp,
    Between,
    CaseWhen,
    DerivedTable,
    Exists,
    SetOp,
    SetQuery,
    ScalarSubquery,
    WindowExpr,
    WindowFunc,
    pretty_print,
    Scan,
    Filter,
    Join,
    GroupBy,
    Having,
    Project,
    Distinct,
    InList,
    InSubquery,
    Like,
    LimitOffset,
    OrderBy,
    SelectItem,
    WhenClause,
    Compare,
    And,
    Or,
    Not,
    JoinType,
)


# ---------------------------------------------------------------------------
# logger system initialization
# ---------------------------------------------------------------------------

def _setup_logging(run_ts: str):
    """
    setup 2 logger:
      detail_logger : detail log written in logs_detail/<ts>.log
      run_logger    : running log written in logs/<ts>.log

    bug detail log will be set up when a new bug found,file named by the timestamp of the bug found, written in logs_bug/<ts>.log
    """
    os.makedirs("logs_detail", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("logs_bug",    exist_ok=True)

    # detail log:without timestamp,only output,readable
    detail_fmt = logging.Formatter("%(message)s")
    detail_logger = logging.getLogger("retorm.detail")
    detail_logger.setLevel(logging.DEBUG)
    dh = logging.FileHandler(f"logs_detail/{run_ts}.log", encoding="utf-8")
    dh.setFormatter(detail_fmt)
    detail_logger.addHandler(dh)
    detail_logger.propagate = False

    # running log:with timestamp
    run_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    run_logger = logging.getLogger("retorm.run")
    run_logger.setLevel(logging.INFO)
    rh = logging.FileHandler(f"logs/{run_ts}.log", encoding="utf-8")
    rh.setFormatter(run_fmt)
    run_logger.addHandler(rh)
    run_logger.propagate = False

    return detail_logger, run_logger


# ---------------------------------------------------------------------------
# output background thread data per 10 seconds
# ---------------------------------------------------------------------------

class _StatsPrinter(threading.Thread):
    """background thread to print running stats every 10 seconds"""

    def __init__(self, stats, run_logger, interval: int = 10):
        super().__init__(daemon=True)
        self.stats      = stats
        self.run_logger = run_logger
        self.interval   = interval
        self._stop_evt  = threading.Event()
        self.start_time = time.time()

    def run(self):
        while not self._stop_evt.wait(self.interval):
            self._log()

    def stop(self):
        self._stop_evt.set()
        self._log()   # write final status when stopping the thread

    def _log(self):
        elapsed = int(time.time() - self.start_time)
        s = self.stats
        total = s.total_queries

        # rate calculations, avoid division by zero
        pass_rate   = f"{s.passed / total * 100:.1f}%" if total else "N/A"
        empty_rate  = f"{s.empty_results / total * 100:.1f}%" if total else "N/A"
        error_rate  = f"{s.errors / total * 100:.1f}%" if total else "N/A"
        bug_total   = s.sql_bugs + s.sql_true_orm_divergences
        bug_rate    = f"{bug_total / total * 100:.1f}%" if total else "N/A"

        # speed:ueries per minute
        qpm = f"{total / elapsed * 60:.1f}" if elapsed > 0 else "N/A"

        lines = [
            f"[running data]  time={elapsed}s  speed={qpm} q/min , total queries  : {total}",
            f"  passed      : {s.passed}  ({pass_rate}),  empty results   : {s.empty_results}  ({empty_rate}),  errors  : {s.errors}  ({error_rate})",
            f"  SQL bug   : {s.sql_bugs},  SQL vs true ORM diff  : {s.sql_true_orm_divergences},  bug total  : {bug_total}  ({bug_rate})",
            f"  query types  : single-table={s.single_table_queries}, join={s.join_queries}, LEFT JOIN={s.left_join_queries}, filter={s.filter_queries}, groupby={s.groupby_queries}, having={s.having_queries}, duplicate projection={s.duplicate_proj_queries}, NULL predicate={s.null_predicate_queries}",
        ]
        lines = _format_run_stats_lines(
            s, elapsed, qpm, pass_rate, empty_rate, error_rate, bug_rate
        )
        for line in lines:
            self.run_logger.info(line)


# ---------------------------------------------------------------------------
# Bug log & report
# ---------------------------------------------------------------------------

@dataclass
class BugReport:
    schema_id:    int
    query_id:     int
    schema:       object          # Schema object, used to regenerate the schema
    ir:           object          # original IR object, used to regenerate the IR
    ir_str:       str
    schema_seed:  int
    query_seed:   int
    table_data:   dict            # original table data, used to regenerate the table
    rows_per_table: int
    use_z3:       bool
    z3_timeout:   int
    ref_vs_sql:   object          # CompareResult
    ref_vs_true_orm: object       # CompareResult
    ref_rows:     list
    sql_rows:     list
    true_orm_rows: list
    sql_vs_true_orm: object = None   # CompareResult
    true_orm_fact_compare: object = None   # CompareResult
    true_orm_facts: object = None
    sql_text:     str = ""        # generated raw SQL text
    true_orm_compiled_sql: str = ""
    error:        Optional[str] = None


@dataclass
class RunStats:
    total_queries: int = 0
    passed:        int = 0
    sql_bugs:      int = 0
    sql_true_orm_divergences: int = 0
    true_orm_fact_mismatches: int = 0
    ref_path_anomalies: int = 0
    true_orm_unsupported: int = 0
    errors:        int = 0
    empty_results: int = 0
    single_table_queries:   int = 0
    join_queries:           int = 0
    filter_queries:         int = 0
    groupby_queries:        int = 0
    having_queries:         int = 0
    duplicate_proj_queries: int = 0
    null_predicate_queries: int = 0
    left_join_queries:      int = 0
    multi_join_queries:     int = 0
    distinct_queries:       int = 0
    orderby_queries:        int = 0
    orderby_agg_queries:    int = 0
    limit_offset_queries:   int = 0
    in_list_predicate_queries: int = 0
    between_predicate_queries: int = 0
    like_predicate_queries: int = 0
    arithmetic_expr_queries: int = 0
    case_when_queries:      int = 0
    window_expr_queries:    int = 0
    derived_table_queries:   int = 0
    subquery_queries:       int = 0
    exists_subquery_queries: int = 0
    in_subquery_queries:    int = 0
    distinct_order_limit_queries: int = 0
    self_join_queries:      int = 0
    set_query_queries:      int = 0
    entity_projection_queries: int = 0
    entity_scalar_mix_queries: int = 0
    left_join_null_queries: int = 0
    left_join_groupby_queries: int = 0
    left_join_having_queries: int = 0
    left_join_right_proj_queries: int = 0
    left_join_right_predicate_queries: int = 0
    single_row_results: int = 0
    multi_row_results: int = 0
    duplicate_row_results: int = 0
    null_containing_results: int = 0
    entity_duplicate_results: int = 0
    left_join_null_extension_results: int = 0
    aggregation_null_results: int = 0
    row_budget_boost_queries: int = 0
    fault_smoke_attempts: int = 0
    fault_smoke_detected: int = 0
    fault_smoke_missed: int = 0
    true_orm_relationship_join_queries: int = 0
    true_orm_explicit_join_queries: int = 0
    true_orm_entity_queries: int = 0
    true_orm_entity_scalar_mix_queries: int = 0
    true_orm_joinedload_queries: int = 0
    true_orm_selectinload_queries: int = 0
    true_orm_relationship_touch_queries: int = 0
    true_orm_self_alias_queries: int = 0
    true_orm_setop_queries: int = 0
    true_orm_scalar_subquery_queries: int = 0
    true_orm_window_queries: int = 0
    true_orm_derived_table_queries: int = 0
    true_orm_limit_subquery_wrap_queries: int = 0
    schema_self_fk_tables: int = 0
    schema_multi_fk_same_target_tables: int = 0
    schema_assoc_like_tables: int = 0
    schema_hub_like_schemas: int = 0
    queries_by_stress_mode: Dict[str, int] = field(default_factory=dict)
    empty_results_by_stress_mode: Dict[str, int] = field(default_factory=dict)
    bug_reports:   List[BugReport] = field(default_factory=list)


def _report_has_actionable_bug(report: BugReport) -> bool:
    """
    A report should only be persisted when it represents a real SQL vs true ORM
    finding or an execution failure. Ref-path mismatches are diagnostic only.
    """
    if report.error:
        return True
    if report.sql_vs_true_orm is not None and not report.sql_vs_true_orm.match:
        return True
    if report.true_orm_fact_compare is not None and not report.true_orm_fact_compare.match:
        return True
    return False


def _get_report_category(report: BugReport) -> tuple[str, str, str]:
    if report.error:
        return "execution_error", "Execution Error", report.error

    ref_vs_sql = report.ref_vs_sql
    ref_vs_true_orm = report.ref_vs_true_orm
    sql_vs_true_orm = report.sql_vs_true_orm
    true_orm_fact_compare = report.true_orm_fact_compare
    sql_mismatch = ref_vs_sql is not None and not ref_vs_sql.match
    true_orm_mismatch = ref_vs_true_orm is not None and not ref_vs_true_orm.match
    sql_true_orm_mismatch = sql_vs_true_orm is not None and not sql_vs_true_orm.match
    true_orm_fact_mismatch = (
        true_orm_fact_compare is not None and not true_orm_fact_compare.match
    )

    if sql_vs_true_orm is None:
        reason_parts = []
        if sql_mismatch:
            reason_parts.append(f"ref_vs_sql: {ref_vs_sql.reason}")
        if true_orm_mismatch:
            reason_parts.append(f"ref_vs_true_orm: {ref_vs_true_orm.reason}")
        if true_orm_fact_mismatch:
            reason_parts.append(f"true_orm_facts: {true_orm_fact_compare.reason}")
        return "ref_path_anomaly", "Ref Path Anomaly", " | ".join(reason_parts)

    if not sql_true_orm_mismatch:
        if true_orm_fact_mismatch:
            return (
                "true_orm_fact_mismatch",
                "True ORM Fact Mismatch",
                f"true_orm_facts: {true_orm_fact_compare.reason}",
            )
        if sql_mismatch or true_orm_mismatch:
            reason_parts = []
            if sql_mismatch:
                reason_parts.append(f"ref_vs_sql: {ref_vs_sql.reason}")
            if true_orm_mismatch:
                reason_parts.append(f"ref_vs_true_orm: {ref_vs_true_orm.reason}")
            return "ref_path_anomaly", "Ref Path Anomaly", " | ".join(reason_parts)
        return "consistent", "Consistent", ""

    reason_parts = [f"sql_vs_true_orm: {sql_vs_true_orm.reason}"]
    if true_orm_fact_mismatch:
        reason_parts.append(f"true_orm_facts: {true_orm_fact_compare.reason}")
    if sql_mismatch:
        reason_parts.append(f"ref_vs_sql: {ref_vs_sql.reason}")
    if true_orm_mismatch:
        reason_parts.append(f"ref_vs_true_orm: {ref_vs_true_orm.reason}")
    return "sql_true_orm_divergence", "SQL vs True ORM Divergence", " | ".join(reason_parts)


def _classify_bug_report(report: BugReport) -> tuple[str, str]:
    """
    Classify a bug report without relying on CompareResult truthiness.
    CompareResult.__bool__ returns ``match``, so boolean checks can invert the
    intended control flow for mismatches.
    """
    _, label, reason = _get_report_category(report)
    return label, reason


def _increment_mode_counter(counter: Dict[str, int], stress_mode: str, amount: int = 1) -> None:
    counter[stress_mode] = counter.get(stress_mode, 0) + amount


def _compare_true_orm_facts(
    sql_rows: list,
    true_orm_result,
    ordered: bool = False,
    strict: bool = False,
) -> CompareResult:
    facts = getattr(true_orm_result, "facts", None)
    if facts is None or not getattr(facts, "entity_pk_columns", None):
        expected_loaded = getattr(facts, "expected_loaded_relationships", {}) if facts else {}
        actual_loaded = getattr(facts, "loaded_relationships", {}) if facts else {}
        if expected_loaded or actual_loaded:
            return _compare_loaded_relationship_facts(expected_loaded, actual_loaded)
        return CompareResult(match=True, reason="no true ORM facts to compare")

    for alias_name, pk_cols in facts.entity_pk_columns.items():
        expected_pk_rows = _extract_sql_entity_pk_rows(sql_rows, alias_name, pk_cols)
        actual_pk_rows = _pk_rows_from_tuples(facts.entity_pks.get(alias_name, []), len(pk_cols))
        pk_cmp = compare_two_paths(
            expected_pk_rows,
            actual_pk_rows,
            f"sql[{alias_name}].entity_pks",
            f"true_orm[{alias_name}].entity_pks",
            ordered=ordered,
            strict=strict,
        )
        if not pk_cmp.match:
            return pk_cmp

        expected_dup_rows = _duplicate_pk_rows(expected_pk_rows)
        actual_dup_rows = _pk_rows_from_tuples(
            facts.duplicate_entity_pks.get(alias_name, []),
            len(pk_cols),
        )
        dup_cmp = compare_two_paths(
            expected_dup_rows,
            actual_dup_rows,
            f"sql[{alias_name}].duplicate_pks",
            f"true_orm[{alias_name}].duplicate_pks",
            ordered=False,
            strict=True,
        )
        if not dup_cmp.match:
            return dup_cmp

    expected_materialized = _expected_materialized_entity_count(sql_rows, facts)
    if expected_materialized != facts.materialized_entity_count:
        return CompareResult(
            match=False,
            reason=(
                "materialized entity count differs: "
                f"expected={expected_materialized}, actual={facts.materialized_entity_count}"
            ),
            details={
                "expected_materialized_entity_count": expected_materialized,
                "actual_materialized_entity_count": facts.materialized_entity_count,
            },
        )

    expected_loaded = getattr(facts, "expected_loaded_relationships", {}) or {}
    actual_loaded = getattr(facts, "loaded_relationships", {}) or {}
    if not expected_loaded and not actual_loaded:
        expected_identity = _expected_identity_map_size(sql_rows, facts)
        if expected_identity != facts.identity_map_size:
            return CompareResult(
                match=False,
                reason=(
                    "identity map size differs: "
                    f"expected={expected_identity}, actual={facts.identity_map_size}"
                ),
                details={
                    "expected_identity_map_size": expected_identity,
                    "actual_identity_map_size": facts.identity_map_size,
                },
            )

    return _compare_loaded_relationship_facts(
        expected_loaded,
        actual_loaded,
        materialized_aliases={
            alias_name
            for alias_name, pk_rows in getattr(facts, "entity_pks", {}).items()
            if pk_rows
        },
    )


def _compare_loaded_relationship_facts(
    expected_loaded: Dict[str, List[str]],
    actual_loaded: Dict[str, List[str]],
    materialized_aliases: Optional[set] = None,
) -> CompareResult:
    aliases = sorted(set(expected_loaded) | set(actual_loaded))
    for alias_name in aliases:
        if materialized_aliases is not None and alias_name not in materialized_aliases:
            continue
        expected = sorted(set(expected_loaded.get(alias_name, [])))
        actual = sorted(set(actual_loaded.get(alias_name, [])))
        if expected != actual:
            return CompareResult(
                match=False,
                reason=(
                    f"loaded relationships differ for {alias_name}: "
                    f"expected={expected}, actual={actual}"
                ),
                details={
                    "alias": alias_name,
                    "expected_loaded_relationships": expected,
                    "actual_loaded_relationships": actual,
                },
            )
    return CompareResult(match=True)


def _extract_sql_entity_pk_rows(
    sql_rows: list,
    alias_name: str,
    pk_cols: Tuple[str, ...],
) -> List[dict]:
    rows = []
    for row in sql_rows:
        values = []
        missing = False
        for col_name in pk_cols:
            value, found = _extract_sql_fact_value(row, alias_name, col_name)
            if not found:
                missing = True
                break
            values.append(value)
        if missing or any(value is None for value in values):
            continue
        rows.append({f"pk_{idx}": value for idx, value in enumerate(values)})
    return rows


def _extract_sql_fact_value(row: dict, alias_name: str, col_name: str) -> tuple[object, bool]:
    direct_candidates = (
        f"{alias_name}.{col_name}",
        f"{alias_name}_{col_name}",
        col_name,
    )
    for candidate in direct_candidates:
        if candidate in row:
            return row[candidate], True

    suffix = f"_{col_name}"
    dotted_suffix = f".{col_name}"
    matching = [
        value
        for key, value in row.items()
        if key == col_name or key.endswith(suffix) or key.endswith(dotted_suffix)
    ]
    if len(matching) == 1:
        return matching[0], True
    return None, False


def _pk_rows_from_tuples(pk_tuples: List[Tuple[object, ...]], width: int) -> List[dict]:
    rows = []
    for values in pk_tuples:
        padded = list(values[:width]) + [None] * max(0, width - len(values))
        rows.append({f"pk_{idx}": padded[idx] for idx in range(width)})
    return rows


def _duplicate_pk_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    duplicates = []
    for row in rows:
        key = tuple(row.get(col_name) for col_name in sorted(row))
        if key in seen:
            duplicates.append(dict(row))
            continue
        seen.add(key)
    return duplicates


def _expected_identity_map_size(sql_rows: list, facts) -> int:
    identities = set()
    entity_tables = getattr(facts, "entity_tables", {})
    entity_pk_columns = getattr(facts, "entity_pk_columns", {})
    for alias_name, pk_cols in entity_pk_columns.items():
        table_name = entity_tables.get(alias_name, alias_name)
        for row in _extract_sql_entity_pk_rows(sql_rows, alias_name, pk_cols):
            key = tuple(row.get(f"pk_{idx}") for idx in range(len(pk_cols)))
            identities.add((table_name, key))
    return len(identities)


def _expected_materialized_entity_count(sql_rows: list, facts) -> int:
    total = 0
    entity_pk_columns = getattr(facts, "entity_pk_columns", {})
    for alias_name, pk_cols in entity_pk_columns.items():
        for row in _extract_sql_entity_pk_rows(sql_rows, alias_name, pk_cols):
            values = [row.get(f"pk_{idx}") for idx in range(len(pk_cols))]
            if any(value is None for value in values):
                continue
            total += 1
    return total

def _persist_bug_report(
    report: BugReport,
    stats: RunStats,
    bug_dir: str,
    bug_idx: int,
    dlog,
) -> int:
    """
    Persist a bug report only when it is actionable.
    This guards against accidental "all matched" bug files while keeping
    execution errors and result mismatches intact.
    """
    if not _report_has_actionable_bug(report):
        warn = (
            "[internal warning] skipped non-bug report: "
            f"schema={report.schema_id + 1}, query={report.query_id + 1}, "
            "all compared paths matched"
        )
        print(warn)
        dlog(warn)
        return bug_idx

    bug_idx += 1
    stats.bug_reports.append(report)
    fpath = _write_bug_file(report, bug_dir, bug_idx)
    fpath_detail = _write_bug_detail(report)
    msg = f"  -> repro script: {fpath}  bug detail: {fpath_detail}"
    print(msg)
    dlog(msg)
    return bug_idx


# ---------------------------------------------------------------------------
# Bug report file generation
# ---------------------------------------------------------------------------

def _write_bug_detail(report: BugReport) -> str:
    """
    Write a detailed bug log to logs_bug/<bug_ts>.log.
    file named by the timestamp of the bug found, written in logs_bug/<ts>.log
    including:
        1. IR tree
        2. generated Raw SQL
        3. ORM API called description
        4. Program code
        5. three way comparison results
    return the path to the generated bug detail log file.
    """
    bug_ts = datetime.now().strftime("%y%m%d_%H%M%S_%f")[:-3]
    if not _report_has_actionable_bug(report):
        raise ValueError("attempted to write a non-actionable bug detail")

    os.makedirs("logs_bug", exist_ok=True)
    fpath_detail = f"logs_bug/{bug_ts}.log"

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    bug_logger = logging.getLogger(f"retorm.bug.{bug_ts}")
    bug_logger.setLevel(logging.DEBUG)
    bh = logging.FileHandler(fpath_detail, encoding="utf-8")
    bh.setFormatter(fmt)
    bug_logger.addHandler(bh)
    bug_logger.propagate = False

    from generator.schema_gen import generate_create_sqls
    from translators.sql import translate as sql_translate

    _, bug_type, reason = _get_report_category(report)
    sep = "=" * 70

    if report.sql_text:
        sql_str = report.sql_text
    else:
        try:
            sql_str = sql_translate(report.ir)
        except Exception as exc:
            sql_str = f"(SQL generation failed: {exc})"

    lines = [
        sep,
        f"BUG #{report.schema_id + 1}-{report.query_id + 1}",
        f"Type        : {bug_type}",
        f"Reason      : {reason}",
        f"Schema seed : {report.schema_seed}",
        f"Query seed  : {report.query_seed}",
        f"Rows/table  : {report.rows_per_table}",
        f"Use Z3      : {report.use_z3}",
        f"Z3 timeout  : {report.z3_timeout}s",
        sep,
        "",
        "Schema SQL:",
    ]
    lines.extend(generate_create_sqls(report.schema))
    lines.extend([
        "",
        "IR:",
        report.ir_str,
        "",
        "Inserted Data:",
    ])
    for table in report.schema.tables:
        rows = report.table_data.get(table.name, [])
        lines.append(f"  {table.name} ({len(rows)} rows): {rows}")

    lines.extend([
        "",
        "Raw SQL:",
        sql_str,
        "",
        "True ORM Pseudocode:",
        _gen_orm_pseudocode(report),
        "",
        "Results:",
        f"  ref ({len(report.ref_rows)} rows): {report.ref_rows}",
        f"  sql ({len(report.sql_rows)} rows): {report.sql_rows}",
        f"  true_orm ({len(report.true_orm_rows)} rows): {report.true_orm_rows}",
    ])

    if report.true_orm_compiled_sql:
        lines.extend([
            "",
            "True ORM compiled SQL:",
            report.true_orm_compiled_sql,
        ])

    if report.true_orm_facts is not None:
        lines.extend([
            "",
            f"True ORM facts: {report.true_orm_facts}",
        ])

    if report.ref_vs_sql is not None and not report.ref_vs_sql.match:
        lines.append(f"ref vs sql mismatch: {report.ref_vs_sql.reason}")
    if report.ref_vs_true_orm is not None and not report.ref_vs_true_orm.match:
        lines.append(f"ref vs true_orm mismatch: {report.ref_vs_true_orm.reason}")
    if report.sql_vs_true_orm is not None and not report.sql_vs_true_orm.match:
        lines.append(f"sql vs true_orm mismatch: {report.sql_vs_true_orm.reason}")
    if report.true_orm_fact_compare is not None and not report.true_orm_fact_compare.match:
        lines.append(f"true_orm facts mismatch: {report.true_orm_fact_compare.reason}")

    lines.append(sep)
    bug_logger.info("\n".join(lines))
    bug_logger.removeHandler(bh)
    bh.close()
    return fpath_detail


def _gen_orm_pseudocode(report: BugReport) -> str:
    """Generate a short pseudocode summary for the ORM path."""
    from ir.nodes import Scan, Filter, Join, GroupBy, Having, Project

    # translate IR to a simplified pseudocode representation of the ORM API calls
    lines = ["    # IR -> SQLAlchemy ORM pseudocode"]
    for line in report.ir_str.split("\n"):
        lines.append("    # " + line)
    return "\n".join(lines)


def _write_bug_file(report: BugReport, bug_dir: str, bug_idx: int) -> str:
    """
    translate a bug into a self-contained Python script that can be run to reproduce the bug.
    return the path to the generated bug script file.
    """
    if not _report_has_actionable_bug(report):
        raise ValueError("attempted to write a non-actionable bug repro script")
    os.makedirs(bug_dir, exist_ok=True)
    fpath = os.path.join(bug_dir, f"bug_{bug_idx:03d}.py")

    # ensure the bug classification
    _, bug_type, reason = _get_report_category(report)

    # generate the SQL create/drop statements for the schema
    from generator.schema_gen import generate_create_sqls, generate_drop_sqls
    create_sqls = generate_create_sqls(report.schema)
    drop_order  = generate_drop_sqls(report.schema)

    # save the inserted table data, schema, and IR for reproduction
    insert_code = _gen_insert_code(report.schema, report.table_data)
    schema_code = _gen_schema_code(report.schema)
    ir_code = _gen_ir_code(report.ir)

    # repro script
    repro_script = textwrap.dedent(f"""\
    #!/usr/bin/env python3
    \"\"\"
    RetORM Bug repro script
    ==================
    Bug type   : {bug_type}
    reason       : {reason}
    Schema ID  : {report.schema_id + 1}
    Query  ID  : {report.query_id + 1}
    Schema seed: {report.schema_seed}
    Query  seed: {report.query_seed}
    rows/table : {report.rows_per_table}
    use_z3     : {report.use_z3}
    z3_timeout : {report.z3_timeout}s

    way to run?
        conda activate retorm
        python {os.path.basename(fpath)}
    \"\"\"

    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from db.connector import (
        init_database, create_tables, drop_tables, insert_rows,
    )
    from generator.schema_gen import Schema, TableSchema, Column, ForeignKey, ColType
    from translators.python_ref     import execute as ref_execute
    from translators.sql            import execute as sql_execute, translate as sql_translate
    from translators.sqlalchemy_true_orm import execute as true_orm_execute, reset_model_cache
    from comparator.compare         import compare_all, print_report
    from ir.nodes import *
    from runner import _query_requires_ordered_compare

    # 1.generate the tables
    init_database()
    drop_tables({drop_order!r})
    create_tables({create_sqls!r})
    reset_model_cache()

    # 2. insert data
    {insert_code}

    # 3. generate IR
    # directly use the IR generated when the bug was discovered, as the transformation after discovery may have caused the issue
    schema = {schema_code}
    ir = {ir_code}

    print("IR tree")
    from ir.nodes import pretty_print
    print(pretty_print(ir))

    # 4. generate SQL
    print("\\ngenerated SQL:")
    print(sql_translate(ir))

    # 5. execute IR in three paths
    ref_rows = ref_execute(ir)
    sql_rows = sql_execute(ir)
    true_orm_rows = true_orm_execute(ir, schema).rows

    print(f"\\nref result ({{len(ref_rows)}} rows: {{ref_rows}}")
    print(f"sql result ({{len(sql_rows)}} rows: {{sql_rows}}")
    print(f"true_orm result ({{len(true_orm_rows)}} rows: {{true_orm_rows}}")

    # 6. compare results
    ordered = _query_requires_ordered_compare(ir)
    ref_vs_sql, ref_vs_true_orm = compare_all(
        ref_rows, sql_rows, true_orm_rows, ordered=ordered
    )
    print_report(ref_vs_sql, ref_vs_true_orm)

    # 7. cleanup
    drop_tables({drop_order!r})
    print("\\nBug reproduction completed.")
    """)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(repro_script)

    return fpath


def _gen_insert_code(schema, table_data: dict) -> str:
    """Generate insert_rows() code for reproducing the captured dataset."""
    lines = []
    for tname in reversed(generate_drop_sqls(schema)):
        rows = table_data.get(tname, [])
        if not rows:
            continue
        table = schema.get_table(tname)
        cols = [c.name for c in table.columns]
        values = [tuple(row[c] for c in cols) for row in rows]
        lines.append(f"insert_rows({tname!r}, {cols!r}, {values!r})")
    return "\n".join(lines) if lines else "pass"


def _gen_schema_code(schema) -> str:
    table_codes = []
    for table in schema.tables:
        col_codes = []
        for col in table.columns:
            col_codes.append(
                "Column("
                f"name={col.name!r}, "
                f"col_type=ColType.{col.col_type.name}, "
                f"nullable={col.nullable!r}, "
                f"is_pk={col.is_pk!r})"
            )
        fk_codes = []
        for fk in table.fks:
            fk_codes.append(
                "ForeignKey("
                f"src_table={fk.src_table!r}, "
                f"src_col={fk.src_col!r}, "
                f"ref_table={fk.ref_table!r}, "
                f"ref_col={fk.ref_col!r})"
            )
        table_codes.append(
            "TableSchema("
            f"name={table.name!r}, "
            f"columns=[{', '.join(col_codes)}], "
            f"fks=[{', '.join(fk_codes)}])"
        )
    return f"Schema(tables=[{', '.join(table_codes)}])"


def _gen_ir_code(node) -> str:
    """Serialize the current IR object into executable Python code."""
    from ir.nodes import (
        Aggregate,
        And,
        ArithExpr,
        ArithOp,
        Between,
        CaseWhen,
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
        Like,
        LimitOffset,
        Not,
        Or,
        OrderBy,
        OrderKey,
        Project,
        ScalarSubquery,
        Scan,
        SelectItem,
        SetOp,
        SetQuery,
        WindowExpr,
        WhenClause,
    )

    if isinstance(node, Scan):
        return f"Scan(table={node.table!r}, alias={node.alias!r})"
    if isinstance(node, DerivedTable):
        return f"DerivedTable(subquery={_gen_ir_code(node.subquery)}, alias={node.alias!r})"
    if isinstance(node, Filter):
        return (
            f"Filter(condition={_gen_ir_code(node.condition)}, "
            f"child={_gen_ir_code(node.child)})"
        )
    if isinstance(node, Join):
        return (
            f"Join(left={_gen_ir_code(node.left)}, "
            f"right={_gen_ir_code(node.right)}, "
            f"on={_gen_ir_code(node.on)}, "
            f"join_type=JoinType.{node.join_type.name})"
        )
    if isinstance(node, GroupBy):
        aggs = ", ".join(_gen_ir_code(agg) for agg in node.aggregates)
        return (
            f"GroupBy(fields=[{', '.join(_gen_ir_code(field) for field in node.fields)}], "
            f"aggregates=[{aggs}], "
            f"child={_gen_ir_code(node.child)})"
        )
    if isinstance(node, Having):
        return (
            f"Having(condition={_gen_ir_code(node.condition)}, "
            f"child={_gen_ir_code(node.child)})"
        )
    if isinstance(node, Project):
        fields = ", ".join(_gen_ir_code(field) for field in node.fields)
        return f"Project(fields=[{fields}], child={_gen_ir_code(node.child)})"
    if isinstance(node, Distinct):
        return f"Distinct(child={_gen_ir_code(node.child)})"
    if isinstance(node, OrderBy):
        keys = ", ".join(_gen_ir_code(key) for key in node.keys)
        return f"OrderBy(keys=[{keys}], child={_gen_ir_code(node.child)})"
    if isinstance(node, LimitOffset):
        return (
            f"LimitOffset(limit={node.limit!r}, offset={node.offset!r}, "
            f"child={_gen_ir_code(node.child)})"
        )
    if isinstance(node, SetQuery):
        return (
            f"SetQuery(left={_gen_ir_code(node.left)}, right={_gen_ir_code(node.right)}, "
            f"op=SetOp.{node.op.name}, all={node.all!r})"
        )
    if isinstance(node, OrderKey):
        return (
            f"OrderKey(field={_gen_ir_code(node.field)}, "
            f"descending={node.descending!r})"
        )
    if isinstance(node, SelectItem):
        return f"SelectItem(expr={_gen_ir_code(node.expr)}, alias={node.alias!r})"
    if isinstance(node, ArithExpr):
        return (
            f"ArithExpr(left={_gen_ir_code(node.left)}, "
            f"op=ArithOp.{node.op.name}, right={_gen_ir_code(node.right)})"
        )
    if isinstance(node, CaseWhen):
        cases = ", ".join(_gen_ir_code(case) for case in node.cases)
        return f"CaseWhen(cases=[{cases}], else_value={_gen_ir_code(node.else_value)})"
    if isinstance(node, ScalarSubquery):
        return f"ScalarSubquery(subquery={_gen_ir_code(node.subquery)})"
    if isinstance(node, WindowExpr):
        partition_sql = ", ".join(_gen_ir_code(item) for item in node.partition_by)
        order_sql = ", ".join(_gen_ir_code(item) for item in node.order_by)
        return (
            f"WindowExpr(func=WindowFunc.{node.func.name}, field={_gen_ir_code(node.field)}, "
            f"partition_by=[{partition_sql}], order_by=[{order_sql}])"
        )
    if isinstance(node, WhenClause):
        return (
            f"WhenClause(condition={_gen_ir_code(node.condition)}, "
            f"value={_gen_ir_code(node.value)})"
        )
    if isinstance(node, Compare):
        return (
            f"Compare(field={_gen_ir_code(node.field)}, op=CmpOp.{node.op.name}, "
            f"value={_gen_ir_code(node.value)})"
        )
    if isinstance(node, InList):
        values = ", ".join(_gen_ir_code(value) for value in node.values)
        return (
            f"InList(field={_gen_ir_code(node.field)}, values=[{values}], "
            f"negated={node.negated!r})"
        )
    if isinstance(node, Between):
        return (
            f"Between(field={_gen_ir_code(node.field)}, lower={_gen_ir_code(node.lower)}, "
            f"upper={_gen_ir_code(node.upper)}, negated={node.negated!r})"
        )
    if isinstance(node, Like):
        return (
            f"Like(field={_gen_ir_code(node.field)}, pattern={node.pattern!r}, "
            f"negated={node.negated!r})"
        )
    if isinstance(node, Exists):
        return f"Exists(subquery={_gen_ir_code(node.subquery)}, negated={node.negated!r})"
    if isinstance(node, InSubquery):
        return (
            f"InSubquery(field={_gen_ir_code(node.field)}, "
            f"subquery={_gen_ir_code(node.subquery)}, negated={node.negated!r})"
        )
    if isinstance(node, And):
        return f"And(left={_gen_ir_code(node.left)}, right={_gen_ir_code(node.right)})"
    if isinstance(node, Or):
        return f"Or(left={_gen_ir_code(node.left)}, right={_gen_ir_code(node.right)})"
    if isinstance(node, Not):
        return f"Not(child={_gen_ir_code(node.child)})"
    if isinstance(node, Aggregate):
        return (
            f"Aggregate(func=AggFunc.{node.func.name}, "
            f"field={_gen_ir_code(node.field)}, alias={node.alias!r})"
        )
    if isinstance(node, str):
        return repr(node)
    if isinstance(node, (int, float)) or node is None:
        return repr(node)
    raise TypeError(f"unhandled IR node: {type(node)}")


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run(
    num_schemas:        int   = config.NUM_SCHEMAS,
    queries_per_schema: int   = config.QUERIES_PER_SCHEMA,
    num_tables:         int   = 2,
    cols_per_table:     int   = 3,
    rows_per_table:     int   = config.RANDOM_ROWS,
    use_z3:             bool  = True,
    seed:               Optional[int] = None,
    verbose:            bool  = False,
    strict_compare:     bool  = False,
) -> RunStats:

    run_ts  = datetime.now().strftime("%y%m%d_%H%M%S")
    bug_dir = os.path.join("bugs", run_ts)

    detail_logger, run_logger = _setup_logging(run_ts)

    def dlog(msg: str):
        """Write the detailed log file without double-printing to the console."""
        detail_logger.info(msg)

    # start the background thread
    stats   = RunStats()
    reset_true_orm_coverage()
    printer = _StatsPrinter(stats, run_logger, interval=10)
    printer.start()

    # start info
    header = (
        f"{'=' * 60}\n"
        f"RetORM differential testing start  [{run_ts}]\n"
        f"  schemas={num_schemas}, queries/schema={queries_per_schema}\n"
        f"  tables={num_tables}, cols={cols_per_table}, base_rows={rows_per_table}\n"
        f"  use_z3={use_z3}, seed={seed}\n"
        f"  strict_compare={strict_compare}\n"
        f"  detail_log=logs_detail/{run_ts}.log\n"
        f"  run_log=logs/{run_ts}.log\n"
        f"{'=' * 60}"
    )
    print(header)
    dlog(header)
    run_logger.info("=" * 50)
    run_logger.info(f"RetORM differential testing start  [{run_ts}]")
    run_logger.info(f"  schemas={num_schemas}  queries/schema={queries_per_schema}")
    run_logger.info(f"  tables={num_tables}  cols/table={cols_per_table}")
    run_logger.info(f"  use_z3={use_z3}  seed={seed}")
    run_logger.info(
        "  row budget  : "
        f"base={rows_per_table}  extra_random={config.EXTRA_RANDOM_ROWS}  "
        f"edge={config.EDGE_ROWS}  adversarial={config.ADVERSARIAL_ROWS}"
    )
    run_logger.info(
        "  fault smoke : "
        f"main_fault={config.TRUE_ORM_FAULT_INJECTION}  sample_rate={config.TRUE_ORM_FAULT_SMOKE_RATE}"
    )
    run_logger.info(f"  Z3_TIMEOUT={config.Z3_TIMEOUT_SEC}s")
    run_logger.info(f"  detail_log : logs_detail/{run_ts}.log")
    run_logger.info(f"  bug_detail : logs_bug/<bug_ts>.log")
    run_logger.info("=" * 50)

    init_database()
    base_seed = seed if seed is not None else 0
    bug_idx   = 0

    try:
        for schema_id in range(num_schemas):
            schema_seed = base_seed + schema_id * 1000
            sep = f"\n{'-' * 50}\nSchema {schema_id + 1}/{num_schemas}  (seed={schema_seed})"
            print(sep)
            dlog(sep)
            run_logger.info(
                f"[Schema {schema_id + 1}/{num_schemas}]  start "
                f"seed={schema_seed}  tables={num_tables}"
            )

            # generate schema
            schema_profile = _choose_schema_profile(stats)
            with _temporary_schema_profile(schema_profile):
                schema = generate_schema(
                    num_tables=num_tables,
                    cols_per_table=cols_per_table,
                    fk_prob=schema_profile["fk_prob"],
                    seed=schema_seed,
                )
            _record_schema_shape(stats, schema)
            if verbose:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    print_schema(schema)
                schema_desc = buf.getvalue()
                print(schema_desc, end="")
                dlog(schema_desc)

            # generate schema
            try:
                drop_tables(generate_drop_sqls(schema))
                create_tables(generate_create_sqls(schema))
                reset_model_cache()
            except Exception as e:
                msg = f"[runner] generate schema fail,skip this schema: {e}"
                print(msg); dlog(msg)
                continue

            # each IR
            for query_id in range(queries_per_schema):
                query_seed = schema_seed + query_id + 1
                stats.total_queries += 1
                table_data = {}
                stress_mode = _choose_stress_mode(query_seed, stats)
                _increment_mode_counter(stats.queries_by_stress_mode, stress_mode)

                prefix = (
                    f"\n  Query {query_id + 1}/{queries_per_schema}  "
                    f"(seed={query_seed}, mode={stress_mode})"
                )
                print(prefix, end="  ")
                # 3. generate IR
                try:
                    ir, ctx = generate_ir(schema, stress_mode=stress_mode, seed=query_seed)
                except Exception as e:
                    msg = f"[IR generate fail] {e}"
                    print(msg); dlog(msg)
                    stats.errors += 1
                    continue

                _record_query_shape(stats, ir, ctx)

                if config.ENABLE_TRUE_ORM_PATH:
                    support_ok, support_reason = supports_true_orm(ir)
                    if not support_ok:
                        msg = f"[true_orm unsupported] {support_reason}"
                        print(msg); dlog(msg)
                        stats.true_orm_unsupported += 1
                        continue

                ir_str = pretty_print(ir)
                if verbose:
                    print(f"\n{ir_str}")
                # write query head + IR in one time
                dlog(f"{prefix}\nIR:\n{ir_str}")

                # 4. generate data
                try:
                    _truncate_schema(schema)
                    effective_rows_per_table = _choose_effective_row_budget(
                        rows_per_table,
                        stress_mode,
                        stats,
                    )
                    if effective_rows_per_table > rows_per_table:
                        stats.row_budget_boost_queries += 1
                    table_data = generate_and_insert(
                        schema, ir,
                        rows_per_table=effective_rows_per_table,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        stress_mode=stress_mode,
                        seed=query_seed,
                    )
                except Exception as e:
                    msg = f"[generate data fail] {e}"
                    print(msg); dlog(msg)
                    stats.errors += 1
                    continue

                # 5. execute IR in three paths
                sql_text = ""
                true_orm_cov_delta = {}
                true_orm_result = None
                try:
                    ref_rows = ref_execute(ir) if config.ENABLE_REF_PATH else []
                    sql_rows = sql_execute(ir)
                    cov_before = get_true_orm_coverage_snapshot()
                    with _temporary_true_orm_runtime(stress_mode, ctx):
                        true_orm_result = true_orm_execute(ir, schema) if config.ENABLE_TRUE_ORM_PATH else None
                    cov_after = get_true_orm_coverage_snapshot()
                    true_orm_rows = true_orm_result.rows if true_orm_result is not None else []
                    true_orm_cov_delta = _coverage_delta(cov_before, cov_after)
                    _record_true_orm_query_coverage(stats, true_orm_cov_delta)
                    sql_text = sql_translate(ir)

                    if verbose:
                        exec_info = (
                            f"  ref: {ref_rows}\n"
                            f"  sql: {sql_rows}\n"
                            f"  true_orm: {true_orm_rows}\n"
                            f"  SQL: {sql_text}"
                        )
                        if true_orm_result is not None and true_orm_result.compiled_sql:
                            exec_info += f"\n  true_orm SQL(sample): {true_orm_result.compiled_sql}"
                        print(exec_info)
                        # exec_info write with conclusion after this
                except Exception as e:
                    msg = f"[execute IR fail] {e}"
                    print(msg); dlog(msg)
                    if verbose:
                        tb = traceback.format_exc()
                        print(tb); dlog(tb)
                    stats.errors += 1
                    report = BugReport(
                        schema_id=schema_id, query_id=query_id,
                        schema=schema, ir=ir, ir_str=ir_str,
                        schema_seed=schema_seed, query_seed=query_seed,
                        table_data=table_data,
                        rows_per_table=effective_rows_per_table,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        ref_vs_sql=None, ref_vs_true_orm=None,
                        ref_rows=[], sql_rows=[], true_orm_rows=[],
                        error=str(e),
                    )
                    bug_idx = _persist_bug_report(
                        report, stats, bug_dir, bug_idx, dlog
                    )
                    continue
                    stats.bug_reports.append(report)
                    fpath = _write_bug_file(report, bug_dir, bug_idx)
                    fpath_detail = _write_bug_detail(report)
                    msg2 = f"  repro script: {fpath}  bug detail: {fpath_detail}"
                    print(msg2); dlog(msg2)
                    continue

                # 6. compare results
                try:
                    ordered = _query_requires_ordered_compare(ir)
                    if config.ENABLE_REF_PATH:
                        ref_vs_sql, ref_vs_true_orm = compare_all(
                            ref_rows, sql_rows, true_orm_rows, ordered=ordered, strict=strict_compare
                        )
                    else:
                        ref_vs_sql, ref_vs_true_orm = None, None
                    sql_vs_true_orm = compare_two_paths(
                        sql_rows, true_orm_rows, "sql", "true_orm", ordered=ordered, strict=strict_compare
                    )
                    true_orm_fact_compare = (
                        _compare_true_orm_facts(
                            sql_rows,
                            true_orm_result,
                            ordered=ordered,
                            strict=strict_compare,
                        )
                        if true_orm_result is not None
                        else CompareResult(match=True, reason="true ORM path disabled")
                    )
                except Exception as e:
                    msg = f"[compare results fail] {e}"
                    print(msg); dlog(msg)
                    stats.errors += 1
                    continue

                # 7. record results and report bugs
                all_empty = (len(ref_rows) == 0 and
                             len(sql_rows) == 0 and
                             len(true_orm_rows) == 0)
                outcome_rows = sql_rows if sql_rows else ref_rows
                _record_result_outcomes(stats, outcome_rows, ir, ctx)
                if all_empty:
                    print(" [empty]", end="")
                    stats.empty_results += 1
                    _increment_mode_counter(stats.empty_results_by_stress_mode, stress_mode)

                if sql_vs_true_orm.match and true_orm_fact_compare.match:
                    print(" ok")
                    if verbose:
                        dlog(
                            f"  ref({len(ref_rows)} rows) sql({len(sql_rows)} rows) "
                            f"true_orm({len(true_orm_rows)} rows) -> match"
                            + (" [empty]" if all_empty else "")
                        )
                    else:
                        dlog("  -> match" + (" [empty]" if all_empty else ""))
                    stats.passed += 1
                else:
                    report = BugReport(
                        schema_id=schema_id, query_id=query_id,
                        schema=schema, ir=ir, ir_str=ir_str,
                        schema_seed=schema_seed, query_seed=query_seed,
                        table_data=table_data,
                        rows_per_table=effective_rows_per_table,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        ref_vs_sql=ref_vs_sql, ref_vs_true_orm=ref_vs_true_orm,
                        ref_rows=ref_rows, sql_rows=sql_rows, true_orm_rows=true_orm_rows,
                        sql_vs_true_orm=sql_vs_true_orm,
                        true_orm_fact_compare=true_orm_fact_compare,
                        true_orm_facts=(true_orm_result.facts if true_orm_result is not None else None),
                        sql_text=sql_text,
                        true_orm_compiled_sql=(
                            true_orm_result.compiled_sql if true_orm_result is not None else ""
                        ),
                    )
                    bug_category, tag, reason = _get_report_category(report)
                    if bug_category == "sql_true_orm_divergence":
                        stats.sql_true_orm_divergences += 1
                        tag = "SQL vs true_orm divergence"
                    elif bug_category == "true_orm_fact_mismatch":
                        stats.true_orm_fact_mismatches += 1
                        tag = "True ORM fact mismatch"
                    else:
                        tag = "ref path anomaly"
                        tag = "鈻?ref path anomaly"
                    dlog(
                        f"  ref({len(ref_rows)} rows) sql({len(sql_rows)} rows) "
                        f"true_orm({len(true_orm_rows)} rows) -> {tag}"
                    )

                    # write comparison detail to detail log file
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        print_report(ref_vs_sql, ref_vs_true_orm, ir_str)
                    report_str = buf.getvalue()
                    if verbose:
                        print(report_str)
                    dlog(report_str)
                    if not true_orm_fact_compare.match:
                        dlog(f"  true_orm facts: {true_orm_fact_compare.reason}")

                    bug_idx = _persist_bug_report(
                        report, stats, bug_dir, bug_idx, dlog
                    )
                    continue
                    stats.bug_reports.append(report)
                    fpath = _write_bug_file(report, bug_dir, bug_idx)
                    fpath_detail = _write_bug_detail(report)
                    msg2 = f"  repro script: {fpath}  bug detail: {fpath_detail}"
                    print(msg2); dlog(msg2)

                _maybe_run_fault_smoke(
                    stats,
                    ir,
                    schema,
                    sql_rows,
                    ordered,
                    stress_mode,
                    ctx,
                    dlog,
                    strict_compare,
                )

            # cleanup + schema summary
            try:
                drop_tables(generate_drop_sqls(schema))
            except Exception as e:
                print(f"[runner] clean up fail: {e}")

            # schema level summary written into running log
            schema_total  = queries_per_schema
            schema_bugs   = sum(
                1 for r in stats.bug_reports
                if r.schema_id == schema_id
            )
            schema_errors = sum(
                1 for r in stats.bug_reports
                if r.schema_id == schema_id and r.error
            )
            run_logger.info(
                f"[Schema {schema_id + 1}/{num_schemas}]  end  "
                f"query={schema_total}  bug={schema_bugs}  error={schema_errors}"
            )

    finally:
        printer.stop()

    # final report
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_final_report(stats, bug_dir)
    final_str = buf.getvalue()
    print(final_str)
    dlog(final_str)
    s = stats
    total = s.total_queries
    pass_rate  = f"{s.passed / total * 100:.1f}%" if total else "N/A"
    empty_rate = f"{s.empty_results / total * 100:.1f}%" if total else "N/A"
    run_logger.info("=" * 50)
    run_logger.info(f"RetORM test end  [{run_ts}]")
    run_logger.info(f"  Total queries  : {total}")
    run_logger.info(f"  Passed      : {s.passed}  ({pass_rate})")
    run_logger.info(f"  Empty results   : {s.empty_results}  ({empty_rate})")
    run_logger.info(f"  Errors  : {s.errors}")
    run_logger.info(f"  SQL bugs   : {s.sql_bugs}")
    run_logger.info(f"  SQL vs true ORM diff : {s.sql_true_orm_divergences}")
    run_logger.info(f"  true ORM facts : {s.true_orm_fact_mismatches}")
    run_logger.info(f"  ref anomaly  : {s.ref_path_anomalies}")
    run_logger.info(f"  true_orm unsupported : {s.true_orm_unsupported}")
    run_logger.info(
        f"  Bug count  : {s.sql_bugs + s.sql_true_orm_divergences + s.true_orm_fact_mismatches}"
    )
    run_logger.info(
        "  Struct coverage  : "
        f"Single table={s.single_table_queries}  Join={s.join_queries}  LEFT JOIN={s.left_join_queries}  "
        f"SelfJoin={s.self_join_queries}  SetQuery={s.set_query_queries}  "
        f"EntityProj={s.entity_projection_queries}  Entity+Scalar={s.entity_scalar_mix_queries}  "
        f"Filter={s.filter_queries}  GroupBy={s.groupby_queries}  "
        f"Having={s.having_queries}  Duplicate proj={s.duplicate_proj_queries}  NULL predicate={s.null_predicate_queries}"
    )
    run_logger.info(
        "  schema shapes : "
        f"self_fk_tables={s.schema_self_fk_tables}  multi_fk_target_tables={s.schema_multi_fk_same_target_tables}  "
        f"assoc_like_tables={s.schema_assoc_like_tables}  hub_like_schemas={s.schema_hub_like_schemas}"
    )
    run_logger.info(
        "  true_orm api : "
        f"relationship={s.true_orm_relationship_join_queries}  explicit={s.true_orm_explicit_join_queries}  "
        f"entity={s.true_orm_entity_queries}  entity+scalar={s.true_orm_entity_scalar_mix_queries}  "
        f"joinedload={s.true_orm_joinedload_queries}  selectin={s.true_orm_selectinload_queries}  "
        f"touch={s.true_orm_relationship_touch_queries}  self_alias={s.true_orm_self_alias_queries}  "
        f"setop={s.true_orm_setop_queries}  scalar_subq={s.true_orm_scalar_subquery_queries}  "
        f"window={s.true_orm_window_queries}  derived={s.true_orm_derived_table_queries}"
    )
    run_logger.info(
        "  fault smoke : "
        f"attempts={s.fault_smoke_attempts}  detected={s.fault_smoke_detected}  "
        f"missed={s.fault_smoke_missed}  boosted_rows={s.row_budget_boost_queries}"
    )
    run_logger.info(f"  queries by mode : {_format_mode_counter(s.queries_by_stress_mode)}")
    run_logger.info(f"  empty by mode : {_format_mode_counter(s.empty_results_by_stress_mode)}")
    if s.bug_reports:
        run_logger.info(f"  repro script  : {bug_dir}/")
        run_logger.info(f"  bug report   : logs_bug/<bug_ts>.log")
    run_logger.info("=" * 50)

    dispose_engine()
    return stats


# ---------------------------------------------------------------------------
# auxiliary functions
# ---------------------------------------------------------------------------

def _truncate_schema(schema: Schema) -> None:
    for tname in generate_drop_sqls(schema):
        execute_sql(f"TRUNCATE TABLE `{tname}`;")


@contextmanager
def _temporary_true_orm_runtime(stress_mode: str, ctx) -> None:
    old_strategy = getattr(config, "TRUE_ORM_LOADER_STRATEGY", "off")
    old_touch = getattr(config, "TRUE_ORM_TOUCH_RELATIONSHIPS", False)
    old_join_mode = getattr(config, "TRUE_ORM_JOIN_MODE", "relationship_preferred")

    strategy = old_strategy
    touch = old_touch
    join_mode = old_join_mode
    has_entity_projection = bool(getattr(ctx, "projected_entity_aliases", set()))

    if stress_mode in ("relationship_heavy", "relationship_orderby_heavy"):
        join_mode = "relationship"
    elif stress_mode in ("entity_dedup_heavy", "distinct_entity_heavy", "limit_joined_entity_heavy", "loader_strategy_heavy", "orm_combo_heavy"):
        join_mode = "relationship_preferred"

    if has_entity_projection:
        if stress_mode in ("loader_heavy", "loader_strategy_heavy"):
            strategy = random.choice(["joined", "selectin"])
            touch = True
        elif stress_mode in ("entity_heavy", "entity_dedup_heavy"):
            strategy = "selectin"
        elif stress_mode in ("relationship_heavy", "relationship_orderby_heavy", "limit_joined_entity_heavy", "distinct_entity_heavy", "orm_combo_heavy", "combo_heavy"):
            strategy = "joined"
            touch = random.random() < 0.5

    config.TRUE_ORM_JOIN_MODE = join_mode
    config.TRUE_ORM_LOADER_STRATEGY = strategy
    config.TRUE_ORM_TOUCH_RELATIONSHIPS = touch
    try:
        yield
    finally:
        config.TRUE_ORM_JOIN_MODE = old_join_mode
        config.TRUE_ORM_LOADER_STRATEGY = old_strategy
        config.TRUE_ORM_TOUCH_RELATIONSHIPS = old_touch


def _choose_stress_mode_legacy(query_seed: int) -> str:
    """Legacy deterministic stress mode chooser."""
    rng = random.Random(query_seed ^ 0x5F3759DF)
    roll = rng.random()
    if roll < 0.45:
        return "balanced"
    if roll < 0.65:
        return "join_heavy"
    if roll < 0.8:
        return "groupby_heavy"
    if roll < 0.92:
        return "duplicate_column_heavy"
    return "null_heavy"


def _record_query_shape(stats: RunStats, ir, ctx=None) -> None:
    features = _collect_query_features(ir)
    if ctx is not None:
        if getattr(ctx, "has_self_join", False):
            features["has_self_join"] = True
        if getattr(ctx, "has_set_query", False):
            features["has_set_query"] = True
        if getattr(ctx, "projected_entity_aliases", None):
            features["has_entity_projection"] = True
            projected_fields = getattr(ctx, "projected_fields", []) or []
            entity_cols = sum(
                len(ctx.tables[alias].columns)
                for alias in ctx.projected_entity_aliases
                if alias in ctx.tables
            )
            if len(projected_fields) > entity_cols:
                features["has_entity_scalar_mix"] = True

    if features["has_join"]:
        stats.join_queries += 1
    else:
        stats.single_table_queries += 1
    if features["has_multi_join"]:
        stats.multi_join_queries += 1
    if features["has_left_join"]:
        stats.left_join_queries += 1
    if features["has_distinct"]:
        stats.distinct_queries += 1
    if features["has_orderby"]:
        stats.orderby_queries += 1
    if features["has_orderby_agg"]:
        stats.orderby_agg_queries += 1
    if features["has_limit_offset"]:
        stats.limit_offset_queries += 1
    if features["has_in_list"]:
        stats.in_list_predicate_queries += 1
    if features["has_between"]:
        stats.between_predicate_queries += 1
    if features["has_like"]:
        stats.like_predicate_queries += 1
    if features["has_arithmetic_expr"]:
        stats.arithmetic_expr_queries += 1
    if features["has_case_when"]:
        stats.case_when_queries += 1
    if features["has_window_expr"]:
        stats.window_expr_queries += 1
    if features["has_derived_table"]:
        stats.derived_table_queries += 1
    if features["has_subquery"]:
        stats.subquery_queries += 1
    if features["has_exists_subquery"]:
        stats.exists_subquery_queries += 1
    if features["has_in_subquery"]:
        stats.in_subquery_queries += 1
    if features["has_distinct_order_limit"]:
        stats.distinct_order_limit_queries += 1
    if features["has_self_join"]:
        stats.self_join_queries += 1
    if features["has_set_query"]:
        stats.set_query_queries += 1
    if features["has_entity_projection"]:
        stats.entity_projection_queries += 1
    if features["has_entity_scalar_mix"]:
        stats.entity_scalar_mix_queries += 1
    if features["has_left_join_null"]:
        stats.left_join_null_queries += 1
    if features["has_left_join_groupby"]:
        stats.left_join_groupby_queries += 1
    if features["has_left_join_having"]:
        stats.left_join_having_queries += 1
    if features["has_left_join_right_projection"]:
        stats.left_join_right_proj_queries += 1
    if features["has_left_join_right_predicate"]:
        stats.left_join_right_predicate_queries += 1

    if features["has_filter"]:
        stats.filter_queries += 1
    if features["has_groupby"]:
        stats.groupby_queries += 1
    if features["has_having"]:
        stats.having_queries += 1
    if features["has_duplicate_projection"]:
        stats.duplicate_proj_queries += 1
    if features["has_null_predicate"]:
        stats.null_predicate_queries += 1


def _record_result_outcomes(stats: RunStats, rows, ir, ctx=None) -> None:
    if not rows:
        return

    if len(rows) == 1:
        stats.single_row_results += 1
    elif len(rows) > 1:
        stats.multi_row_results += 1

    if len({_row_signature(row) for row in rows}) < len(rows):
        stats.duplicate_row_results += 1

    if any(any(value is None for value in row.values()) for row in rows):
        stats.null_containing_results += 1

    features = _collect_query_features(ir)
    if features["has_left_join"] and _rows_have_left_join_null_extension(rows, ctx):
        stats.left_join_null_extension_results += 1

    agg_aliases = set(getattr(ctx, "agg_aliases", []) or [])
    if agg_aliases and any(any(row.get(alias) is None for alias in agg_aliases) for row in rows):
        stats.aggregation_null_results += 1

    if getattr(ctx, "projected_entity_aliases", None) and _rows_have_entity_duplicates(rows, ctx):
        stats.entity_duplicate_results += 1


def _rows_have_left_join_null_extension(rows, ctx=None) -> bool:
    right_aliases = set(getattr(ctx, "left_join_right_aliases", set()) or set())
    if not right_aliases:
        return False
    for row in rows:
        for key, value in row.items():
            if "." not in key:
                continue
            alias_name, _ = key.split(".", 1)
            if alias_name in right_aliases and value is None:
                return True
    return False


def _rows_have_entity_duplicates(rows, ctx=None) -> bool:
    entity_aliases = list(getattr(ctx, "projected_entity_aliases", set()) or [])
    if not entity_aliases:
        return False
    for alias_name in entity_aliases:
        seen = set()
        alias_keys = sorted({key for row in rows for key in row.keys() if key.startswith(f"{alias_name}.")})
        if not alias_keys:
            continue
        for row in rows:
            entity_key = tuple((key, _normalize_hashable(row.get(key))) for key in alias_keys)
            if entity_key in seen:
                return True
            seen.add(entity_key)
    return False


def _row_signature(row) -> tuple:
    return tuple(sorted((key, _normalize_hashable(value)) for key, value in row.items()))


def _normalize_hashable(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _record_schema_shape(stats: RunStats, schema: Schema) -> None:
    inbound: Dict[str, int] = {}
    for table in schema.tables:
        target_counts: Dict[str, int] = {}
        for fk in table.fks:
            inbound[fk.ref_table] = inbound.get(fk.ref_table, 0) + 1
            target_counts[fk.ref_table] = target_counts.get(fk.ref_table, 0) + 1
            if fk.ref_table == table.name:
                stats.schema_self_fk_tables += 1
        if any(count >= 2 for count in target_counts.values()):
            stats.schema_multi_fk_same_target_tables += 1
        if len({fk.ref_table for fk in table.fks}) >= 2:
            stats.schema_assoc_like_tables += 1
    if any(count >= 2 for count in inbound.values()):
        stats.schema_hub_like_schemas += 1


def _coverage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    keys = set(before) | set(after)
    return {key: max(0, after.get(key, 0) - before.get(key, 0)) for key in keys}


def _record_true_orm_query_coverage(stats: RunStats, delta: Dict[str, int]) -> None:
    if delta.get("relationship_join_used", 0):
        stats.true_orm_relationship_join_queries += 1
    if delta.get("explicit_join_used", 0):
        stats.true_orm_explicit_join_queries += 1
    if delta.get("entity_materialization_used", 0):
        stats.true_orm_entity_queries += 1
    if delta.get("entity_scalar_mix_used", 0):
        stats.true_orm_entity_scalar_mix_queries += 1
    if delta.get("joinedload_used", 0):
        stats.true_orm_joinedload_queries += 1
    if delta.get("selectinload_used", 0):
        stats.true_orm_selectinload_queries += 1
    if delta.get("relationship_touch_used", 0):
        stats.true_orm_relationship_touch_queries += 1
    if delta.get("self_alias_used", 0):
        stats.true_orm_self_alias_queries += 1
    if delta.get("set_query_used", 0):
        stats.true_orm_setop_queries += 1
    if delta.get("scalar_subquery_used", 0):
        stats.true_orm_scalar_subquery_queries += 1
    if delta.get("window_expr_used", 0):
        stats.true_orm_window_queries += 1
    if delta.get("derived_table_used", 0):
        stats.true_orm_derived_table_queries += 1
    if delta.get("limit_in_subquery_wrap_used", 0):
        stats.true_orm_limit_subquery_wrap_queries += 1


def _choose_effective_row_budget(
    base_rows: int,
    stress_mode: str,
    stats: Optional[RunStats] = None,
) -> int:
    if stats is None or stats.total_queries < 12:
        return base_rows

    total = max(1, stats.total_queries)
    rows = base_rows
    mode_queries = stats.queries_by_stress_mode.get(stress_mode, 0)
    mode_empty = stats.empty_results_by_stress_mode.get(stress_mode, 0)
    mode_empty_rate = (mode_empty / mode_queries) if mode_queries else 0.0
    overall_empty_rate = stats.empty_results / total

    if stats.multi_row_results / total < 0.28:
        rows += 2
    if mode_queries >= 6:
        if mode_empty_rate >= 0.5:
            rows += 4
        elif mode_empty_rate >= 0.38:
            rows += 2
    elif overall_empty_rate >= 0.36 and stress_mode in (
        "join_heavy",
        "groupby_heavy",
        "subquery_heavy",
        "relationship_heavy",
        "relationship_orderby_heavy",
        "combo_heavy",
    ):
        rows += 1
    if stress_mode in ("join_heavy", "relationship_heavy", "relationship_orderby_heavy", "null_heavy"):
        if stats.left_join_null_extension_results / total < 0.04:
            rows += 2
    if stress_mode in ("groupby_heavy", "combo_heavy", "subquery_heavy", "orderby_heavy", "derived_heavy", "window_heavy"):
        if stats.aggregation_null_results / total < 0.05:
            rows += 1
    if stress_mode in (
        "entity_heavy",
        "entity_dedup_heavy",
        "distinct_entity_heavy",
        "limit_joined_entity_heavy",
        "loader_heavy",
        "loader_strategy_heavy",
        "orm_combo_heavy",
    ):
        if stats.entity_duplicate_results / total < 0.06:
            rows += 2
        if stats.duplicate_row_results / total < 0.14:
            rows += 2

    return max(base_rows, rows)


def _choose_schema_profile(stats: Optional[RunStats] = None) -> Dict[str, float]:
    profile = {
        "fk_prob": config.SCHEMA_FK_PROB,
        "nullable_fk_prob": config.SCHEMA_NULLABLE_FK_PROB,
        "extra_fk_prob": config.SCHEMA_EXTRA_FK_PROB,
        "self_fk_prob": config.SCHEMA_SELF_FK_PROB,
        "assoc_table_prob": config.SCHEMA_ASSOC_TABLE_PROB,
        "multi_fk_same_target_prob": config.SCHEMA_MULTI_FK_SAME_TARGET_PROB,
        "hub_table_prob": config.SCHEMA_HUB_TABLE_PROB,
        "backlink_fk_prob": config.SCHEMA_BACKLINK_FK_PROB,
    }
    if stats is None or stats.total_queries < 10:
        return profile

    total = max(1, stats.total_queries)
    orm_cov = get_true_orm_coverage_snapshot()

    def boost(value: float, delta: float) -> float:
        return max(0.0, min(0.95, value + delta))

    if stats.join_queries / total < 0.5:
        profile["fk_prob"] = boost(profile["fk_prob"], 0.12)
        profile["extra_fk_prob"] = boost(profile["extra_fk_prob"], 0.18)
    if stats.left_join_queries / total < 0.08 or stats.left_join_null_extension_results / total < 0.04:
        profile["nullable_fk_prob"] = boost(profile["nullable_fk_prob"], 0.18)
    if stats.self_join_queries / total < 0.04:
        profile["self_fk_prob"] = boost(profile["self_fk_prob"], 0.2)
    if orm_cov.get("relationship_join_used", 0) / total < 0.08:
        profile["multi_fk_same_target_prob"] = boost(profile["multi_fk_same_target_prob"], 0.2)
        profile["hub_table_prob"] = boost(profile["hub_table_prob"], 0.16)
        profile["backlink_fk_prob"] = boost(profile["backlink_fk_prob"], 0.14)
    if stats.multi_join_queries / total < 0.12:
        profile["assoc_table_prob"] = boost(profile["assoc_table_prob"], 0.14)
        profile["hub_table_prob"] = boost(profile["hub_table_prob"], 0.12)

    return profile


@contextmanager
def _temporary_schema_profile(profile: Dict[str, float]):
    old_nullable = config.SCHEMA_NULLABLE_FK_PROB
    old_extra = config.SCHEMA_EXTRA_FK_PROB
    old_self = config.SCHEMA_SELF_FK_PROB
    old_assoc = config.SCHEMA_ASSOC_TABLE_PROB
    old_multi = config.SCHEMA_MULTI_FK_SAME_TARGET_PROB
    old_hub = config.SCHEMA_HUB_TABLE_PROB
    old_backlink = config.SCHEMA_BACKLINK_FK_PROB

    config.SCHEMA_NULLABLE_FK_PROB = profile["nullable_fk_prob"]
    config.SCHEMA_EXTRA_FK_PROB = profile["extra_fk_prob"]
    config.SCHEMA_SELF_FK_PROB = profile["self_fk_prob"]
    config.SCHEMA_ASSOC_TABLE_PROB = profile["assoc_table_prob"]
    config.SCHEMA_MULTI_FK_SAME_TARGET_PROB = profile["multi_fk_same_target_prob"]
    config.SCHEMA_HUB_TABLE_PROB = profile["hub_table_prob"]
    config.SCHEMA_BACKLINK_FK_PROB = profile["backlink_fk_prob"]
    try:
        yield
    finally:
        config.SCHEMA_NULLABLE_FK_PROB = old_nullable
        config.SCHEMA_EXTRA_FK_PROB = old_extra
        config.SCHEMA_SELF_FK_PROB = old_self
        config.SCHEMA_ASSOC_TABLE_PROB = old_assoc
        config.SCHEMA_MULTI_FK_SAME_TARGET_PROB = old_multi
        config.SCHEMA_HUB_TABLE_PROB = old_hub
        config.SCHEMA_BACKLINK_FK_PROB = old_backlink


def _collect_query_features(node) -> dict:
    features = {
        "has_join": False,
        "has_multi_join": False,
        "has_filter": False,
        "has_groupby": False,
        "has_having": False,
        "has_duplicate_projection": False,
        "has_null_predicate": False,
        "has_left_join": False,
        "has_distinct": False,
        "has_orderby": False,
        "has_orderby_agg": False,
        "has_limit_offset": False,
        "has_in_list": False,
        "has_between": False,
        "has_like": False,
        "has_arithmetic_expr": False,
        "has_case_when": False,
        "has_window_expr": False,
        "has_derived_table": False,
        "has_subquery": False,
        "has_exists_subquery": False,
        "has_in_subquery": False,
        "has_distinct_order_limit": False,
        "has_self_join": False,
        "has_set_query": False,
        "has_entity_projection": False,
        "has_entity_scalar_mix": False,
        "has_left_join_null": False,
        "has_left_join_groupby": False,
        "has_left_join_having": False,
        "has_left_join_right_projection": False,
        "has_left_join_right_predicate": False,
    }

    join_count = 0
    left_join_right_aliases = set()

    def collect_aliases(cur):
        if isinstance(cur, Scan):
            return {cur.alias}
        if isinstance(cur, DerivedTable):
            return {cur.alias}
        if isinstance(cur, Join):
            return collect_aliases(cur.left) | collect_aliases(cur.right)
        if isinstance(cur, Filter):
            return collect_aliases(cur.child)
        if isinstance(cur, GroupBy):
            return collect_aliases(cur.child)
        if isinstance(cur, Having):
            return collect_aliases(cur.child)
        if isinstance(cur, Project):
            return collect_aliases(cur.child)
        if isinstance(cur, Distinct):
            return collect_aliases(cur.child)
        if isinstance(cur, OrderBy):
            return collect_aliases(cur.child)
        if isinstance(cur, LimitOffset):
            return collect_aliases(cur.child)
        if isinstance(cur, SetQuery):
            return collect_aliases(cur.left) | collect_aliases(cur.right)
        return set()

    def field_uses_left_join_right(field_name: str) -> bool:
        if not isinstance(field_name, str) or "." not in field_name:
            return False
        return field_name.split(".", 1)[0] in left_join_right_aliases

    def project_field_name(field) -> str:
        if isinstance(field, SelectItem):
            return field.alias
        return str(field)

    def visit_expr(expr):
        if isinstance(expr, ArithExpr):
            features["has_arithmetic_expr"] = True
            visit_expr(expr.left)
            visit_expr(expr.right)
            return
        if isinstance(expr, CaseWhen):
            features["has_case_when"] = True
            for case_item in expr.cases:
                visit_condition(case_item.condition, track_right_usage=True)
                visit_expr(case_item.value)
            visit_expr(expr.else_value)
            return
        if isinstance(expr, ScalarSubquery):
            features["has_subquery"] = True
            visit(expr.subquery)
            return
        if isinstance(expr, WindowExpr):
            features["has_window_expr"] = True
            if expr.field not in (None, "*"):
                visit_expr(expr.field)
            for item in expr.partition_by:
                visit_expr(item)
            for key in expr.order_by:
                visit_expr(key.field)
            return

    def visit_condition(cond, track_right_usage=False):
        if isinstance(cond, Compare):
            if cond.value is None:
                features["has_null_predicate"] = True
            if track_right_usage and field_uses_left_join_right(cond.field):
                features["has_left_join_right_predicate"] = True
            visit_expr(cond.field)
            visit_expr(cond.value)
            return
        if isinstance(cond, InList):
            features["has_in_list"] = True
            if track_right_usage and field_uses_left_join_right(cond.field):
                features["has_left_join_right_predicate"] = True
            visit_expr(cond.field)
            for value in cond.values:
                visit_expr(value)
            return
        if isinstance(cond, Between):
            features["has_between"] = True
            if track_right_usage and field_uses_left_join_right(cond.field):
                features["has_left_join_right_predicate"] = True
            visit_expr(cond.field)
            visit_expr(cond.lower)
            visit_expr(cond.upper)
            return
        if isinstance(cond, Like):
            features["has_like"] = True
            if track_right_usage and field_uses_left_join_right(cond.field):
                features["has_left_join_right_predicate"] = True
            visit_expr(cond.field)
            return
        if isinstance(cond, Exists):
            features["has_subquery"] = True
            features["has_exists_subquery"] = True
            visit(cond.subquery)
            return
        if isinstance(cond, InSubquery):
            features["has_subquery"] = True
            features["has_in_subquery"] = True
            visit_expr(cond.field)
            visit(cond.subquery)
            return
        if isinstance(cond, And) or isinstance(cond, Or):
            visit_condition(cond.left, track_right_usage)
            visit_condition(cond.right, track_right_usage)
            return
        if isinstance(cond, Not):
            visit_condition(cond.child, track_right_usage)

    def visit(cur):
        nonlocal join_count
        if isinstance(cur, DerivedTable):
            features["has_derived_table"] = True
            visit(cur.subquery)
            return
        if isinstance(cur, Join):
            join_count += 1
            visit_condition(cur.on, track_right_usage=False)
            features["has_join"] = True
            if cur.join_type == JoinType.LEFT:
                features["has_left_join"] = True
                left_join_right_aliases.update(collect_aliases(cur.right))
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, Filter):
            features["has_filter"] = True
            visit(cur.child)
            visit_condition(cur.condition, track_right_usage=True)
            return
        if isinstance(cur, GroupBy):
            features["has_groupby"] = True
            visit(cur.child)
            for field in cur.fields:
                visit_expr(field)
            for agg in cur.aggregates:
                visit_expr(agg.field)
            return
        if isinstance(cur, Having):
            features["has_having"] = True
            visit(cur.child)
            visit_condition(cur.condition, track_right_usage=True)
            return
        if isinstance(cur, Project):
            visit(cur.child)
            short_names = [_short_field_name(project_field_name(field)) for field in cur.fields]
            features["has_duplicate_projection"] = len(short_names) != len(set(short_names))
            if any(
                isinstance(field, str) and field_uses_left_join_right(field)
                for field in cur.fields
            ):
                features["has_left_join_right_projection"] = True
            for field in cur.fields:
                if isinstance(field, SelectItem):
                    visit_expr(field.expr)
            return
        if isinstance(cur, Distinct):
            features["has_distinct"] = True
            visit(cur.child)
            return
        if isinstance(cur, OrderBy):
            features["has_orderby"] = True
            if any(isinstance(key.field, str) and "." not in key.field for key in cur.keys):
                features["has_orderby_agg"] = True
            if any(isinstance(key.field, str) and field_uses_left_join_right(key.field) for key in cur.keys):
                features["has_left_join_right_projection"] = True
            for key in cur.keys:
                visit_expr(key.field)
            visit(cur.child)
            return
        if isinstance(cur, LimitOffset):
            features["has_limit_offset"] = True
            visit(cur.child)
            return
        if isinstance(cur, SetQuery):
            features["has_distinct"] = True
            features["has_set_query"] = True
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, Scan):
            return

    visit(node)
    features["has_multi_join"] = join_count >= 2
    features["has_left_join_null"] = features["has_left_join"] and features["has_null_predicate"]
    features["has_left_join_groupby"] = features["has_left_join"] and features["has_groupby"]
    features["has_left_join_having"] = features["has_left_join"] and features["has_having"]
    features["has_distinct_order_limit"] = (
        features["has_distinct"] and features["has_orderby"] and features["has_limit_offset"]
    )
    return features


def _short_field_name(field_name: str) -> str:
    return field_name.split(".", 1)[1] if "." in field_name else field_name


def _query_requires_ordered_compare(node) -> bool:
    cur = node
    while cur is not None:
        if isinstance(cur, OrderBy):
            return True
        if isinstance(cur, LimitOffset):
            cur = cur.child
            continue
        if isinstance(cur, (Project, Distinct, Filter, GroupBy, Having)):
            cur = cur.child
            continue
        # Only a final outer ORDER BY makes whole-result ordering observable.
        # ORDER BY inside UNION branches / derived tables should not force
        # ordered comparison for the outer result set.
        return False
    return False


def _format_true_orm_coverage_summary(snapshot: Dict[str, int]) -> str:
    ordered_keys = (
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
    return ", ".join(f"{key}={snapshot.get(key, 0)}" for key in ordered_keys)


def _format_mode_counter(counter: Dict[str, int]) -> str:
    if not counter:
        return "-"
    parts = [f"{mode}={count}" for mode, count in sorted(counter.items())]
    return ", ".join(parts)


def _format_run_stats_lines(
    stats: RunStats,
    elapsed: int,
    qpm: str,
    pass_rate: str,
    empty_rate: str,
    error_rate: str,
    bug_rate: str,
) -> List[str]:
    bug_total = (
        stats.sql_bugs
        + stats.sql_true_orm_divergences
        + stats.true_orm_fact_mismatches
    )
    return [
        f"[running log]  time={elapsed}s  speed={qpm} q/min , query  : {stats.total_queries}",
        f"  passed      : {stats.passed}  ({pass_rate}),  empty results   : {stats.empty_results}  ({empty_rate}),  errors  : {stats.errors}  ({error_rate})",
        f"  SQL bug   : {stats.sql_bugs},  SQL vs true ORM diff  : {stats.sql_true_orm_divergences},  true ORM facts  : {stats.true_orm_fact_mismatches},  ref anomaly  : {stats.ref_path_anomalies},  true_orm unsupported  : {stats.true_orm_unsupported},  bug total  : {bug_total}  ({bug_rate})",
        f"  structure coverage  : single table={stats.single_table_queries}, join={stats.join_queries}, multi join={stats.multi_join_queries}, self join={stats.self_join_queries}, left join={stats.left_join_queries}, filter={stats.filter_queries}, group by={stats.groupby_queries}, having={stats.having_queries}, distinct={stats.distinct_queries}, order by={stats.orderby_queries}, order by agg={stats.orderby_agg_queries}, set query={stats.set_query_queries}, entity proj={stats.entity_projection_queries}, entity+scalar={stats.entity_scalar_mix_queries}, duplicate projection={stats.duplicate_proj_queries}, null predicate={stats.null_predicate_queries}",
        f"  syntax coverage  : limit/offset={stats.limit_offset_queries}, IN={stats.in_list_predicate_queries}, BETWEEN={stats.between_predicate_queries}, LIKE={stats.like_predicate_queries}, arithmetic={stats.arithmetic_expr_queries}, CASE={stats.case_when_queries}, window={stats.window_expr_queries}, derived={stats.derived_table_queries}, subquery={stats.subquery_queries}, EXISTS={stats.exists_subquery_queries}, IN-subquery={stats.in_subquery_queries}, distinct+order+limit={stats.distinct_order_limit_queries}",
        f"  left join combinations  : LEFT+NULL={stats.left_join_null_queries}, LEFT+GroupBy={stats.left_join_groupby_queries}, LEFT+Having={stats.left_join_having_queries}, LEFT+right projection={stats.left_join_right_proj_queries}, LEFT+right predicate={stats.left_join_right_predicate_queries}",
        f"  outcome coverage  : single-row={stats.single_row_results}, multi-row={stats.multi_row_results}, duplicate-rows={stats.duplicate_row_results}, null-rows={stats.null_containing_results}, entity-dup={stats.entity_duplicate_results}, left-null-ext={stats.left_join_null_extension_results}, agg-null={stats.aggregation_null_results}",
        f"  schema shapes  : self-fk tables={stats.schema_self_fk_tables}, multi-fk-target tables={stats.schema_multi_fk_same_target_tables}, assoc-like tables={stats.schema_assoc_like_tables}, hub-like schemas={stats.schema_hub_like_schemas}",
        f"  true_orm api  : relationship={stats.true_orm_relationship_join_queries}, explicit={stats.true_orm_explicit_join_queries}, entity={stats.true_orm_entity_queries}, entity+scalar={stats.true_orm_entity_scalar_mix_queries}, joinedload={stats.true_orm_joinedload_queries}, selectin={stats.true_orm_selectinload_queries}, touch-rel={stats.true_orm_relationship_touch_queries}, self-alias={stats.true_orm_self_alias_queries}, setop={stats.true_orm_setop_queries}, scalar-subq={stats.true_orm_scalar_subquery_queries}, window={stats.true_orm_window_queries}, derived={stats.true_orm_derived_table_queries}, in-limit-wrap={stats.true_orm_limit_subquery_wrap_queries}",
        f"  fault smoke  : attempts={stats.fault_smoke_attempts}, detected={stats.fault_smoke_detected}, missed={stats.fault_smoke_missed}, boosted-rows={stats.row_budget_boost_queries}",
        f"  stress modes  : {_format_mode_counter(stats.queries_by_stress_mode)}",
        f"  empty by mode : {_format_mode_counter(stats.empty_results_by_stress_mode)}",
        f"  true_orm path  : {_format_true_orm_coverage_summary(get_true_orm_coverage_snapshot())}",
    ]


def _query_has_positive_offset(node) -> bool:
    if isinstance(node, LimitOffset):
        return node.offset > 0 or _query_has_positive_offset(node.child)
    if isinstance(node, DerivedTable):
        return _query_has_positive_offset(node.subquery)
    if isinstance(node, SetQuery):
        return _query_has_positive_offset(node.left) or _query_has_positive_offset(node.right)
    if isinstance(node, Join):
        return _query_has_positive_offset(node.left) or _query_has_positive_offset(node.right)
    if hasattr(node, "child"):
        return _query_has_positive_offset(node.child)
    return False


def _query_has_count_aggregate(node) -> bool:
    if isinstance(node, GroupBy):
        if any(agg.func.value == "COUNT" and agg.field != "*" for agg in node.aggregates):
            return True
        return _query_has_count_aggregate(node.child)
    if isinstance(node, DerivedTable):
        return _query_has_count_aggregate(node.subquery)
    if isinstance(node, SetQuery):
        return _query_has_count_aggregate(node.left) or _query_has_count_aggregate(node.right)
    if isinstance(node, Join):
        return _query_has_count_aggregate(node.left) or _query_has_count_aggregate(node.right)
    if hasattr(node, "child"):
        return _query_has_count_aggregate(node.child)
    return False


def _query_has_null_compare(node) -> bool:
    return _collect_query_features(node)["has_null_predicate"]


def _candidate_fault_modes(node) -> List[str]:
    features = _collect_query_features(node)
    candidates: List[str] = []
    if features["has_left_join"]:
        candidates.append("inner_for_left_join")
    if features["has_orderby"]:
        candidates.append("reverse_order")
    if _query_has_positive_offset(node):
        candidates.append("drop_offset")
    if _query_has_count_aggregate(node):
        candidates.append("count_star")
    if _query_has_null_compare(node):
        candidates.append("null_eq_false")
    return candidates


@contextmanager
def _temporary_true_orm_fault_mode(mode: str):
    old_mode = getattr(config, "TRUE_ORM_FAULT_INJECTION", "off")
    config.TRUE_ORM_FAULT_INJECTION = mode
    try:
        yield
    finally:
        config.TRUE_ORM_FAULT_INJECTION = old_mode


def _maybe_run_fault_smoke(
    stats: RunStats,
    ir,
    schema: Schema,
    sql_rows: list,
    ordered: bool,
    stress_mode: str,
    ctx,
    dlog,
    strict_compare: bool = False,
) -> None:
    rate = max(0.0, float(getattr(config, "TRUE_ORM_FAULT_SMOKE_RATE", 0.0) or 0.0))
    if rate <= 0.0 or random.random() > rate:
        return
    if getattr(config, "TRUE_ORM_FAULT_INJECTION", "off") not in ("", "off", None):
        return

    candidates = _candidate_fault_modes(ir)
    if not candidates:
        return

    mode = random.choice(candidates)
    stats.fault_smoke_attempts += 1
    try:
        with _temporary_true_orm_runtime(stress_mode, ctx):
            with _temporary_true_orm_fault_mode(mode):
                faulty_rows = true_orm_execute(ir, schema).rows
        smoke_cmp = compare_two_paths(
            sql_rows,
            faulty_rows,
            "sql",
            f"faulty_true_orm[{mode}]",
            ordered=ordered,
            strict=strict_compare,
        )
        if smoke_cmp.match:
            stats.fault_smoke_missed += 1
            dlog(f"[fault smoke missed] mode={mode}")
        else:
            stats.fault_smoke_detected += 1
            dlog(f"[fault smoke caught] mode={mode} reason={smoke_cmp.reason}")
    except Exception as exc:
        stats.fault_smoke_detected += 1
        dlog(f"[fault smoke caught] mode={mode} raised={exc}")


def _print_final_report(stats: RunStats, bug_dir: str = "bugs") -> None:
    print("\n" + "=" * 60)
    print("Test Complete")
    print(f"  total queries : {stats.total_queries}")
    print(f"  passed        : {stats.passed}")
    print(f"  empty results : {stats.empty_results}")
    print(f"  execution err : {stats.errors}")
    print(f"  SQL bugs      : {stats.sql_bugs}")
    print(f"  SQL vs true ORM diff : {stats.sql_true_orm_divergences}")
    print(f"  true ORM facts: {stats.true_orm_fact_mismatches}")
    print(f"  ref anomaly   : {stats.ref_path_anomalies}")
    print(f"  true_orm unsupported : {stats.true_orm_unsupported}")
    print(f"  bug total     : {stats.sql_bugs + stats.sql_true_orm_divergences + stats.true_orm_fact_mismatches}")
    print("  Structure Coverage:")
    print(f"    single table : {stats.single_table_queries}")
    print(f"    join         : {stats.join_queries}")
    print(f"    multi join   : {stats.multi_join_queries}")
    print(f"    self join    : {stats.self_join_queries}")
    print(f"    left join    : {stats.left_join_queries}")
    print(f"    filter       : {stats.filter_queries}")
    print(f"    group by     : {stats.groupby_queries}")
    print(f"    having       : {stats.having_queries}")
    print(f"    distinct     : {stats.distinct_queries}")
    print(f"    order by     : {stats.orderby_queries}")
    print(f"    order by agg : {stats.orderby_agg_queries}")
    print(f"    limit/offset : {stats.limit_offset_queries}")
    print(f"    IN           : {stats.in_list_predicate_queries}")
    print(f"    BETWEEN      : {stats.between_predicate_queries}")
    print(f"    LIKE         : {stats.like_predicate_queries}")
    print(f"    arithmetic   : {stats.arithmetic_expr_queries}")
    print(f"    CASE         : {stats.case_when_queries}")
    print(f"    window       : {stats.window_expr_queries}")
    print(f"    derived      : {stats.derived_table_queries}")
    print(f"    subquery     : {stats.subquery_queries}")
    print(f"    EXISTS       : {stats.exists_subquery_queries}")
    print(f"    IN-subquery  : {stats.in_subquery_queries}")
    print(f"    set query    : {stats.set_query_queries}")
    print(f"    entity proj  : {stats.entity_projection_queries}")
    print(f"    entity+scalar: {stats.entity_scalar_mix_queries}")
    print(f"    duplicate proj: {stats.duplicate_proj_queries}")
    print(f"    null predicate: {stats.null_predicate_queries}")
    print("  Schema Shapes:")
    print(f"    self-fk tables        : {stats.schema_self_fk_tables}")
    print(f"    multi-fk target tables: {stats.schema_multi_fk_same_target_tables}")
    print(f"    assoc-like tables     : {stats.schema_assoc_like_tables}")
    print(f"    hub-like schemas      : {stats.schema_hub_like_schemas}")
    print(f"  queries by mode : {_format_mode_counter(stats.queries_by_stress_mode)}")
    print(f"  empty by mode   : {_format_mode_counter(stats.empty_results_by_stress_mode)}")
    print("  Left Join Combos:")
    print(f"    LEFT+NULL : {stats.left_join_null_queries}")
    print(f"    LEFT+GB   : {stats.left_join_groupby_queries}")
    print(f"    LEFT+HAV  : {stats.left_join_having_queries}")
    print(f"    LEFT+Proj : {stats.left_join_right_proj_queries}")
    print(f"    LEFT+Pred : {stats.left_join_right_predicate_queries}")
    print("  True ORM API Coverage:")
    print(f"    relationship joins : {stats.true_orm_relationship_join_queries}")
    print(f"    explicit joins     : {stats.true_orm_explicit_join_queries}")
    print(f"    entity materialize : {stats.true_orm_entity_queries}")
    print(f"    entity+scalar      : {stats.true_orm_entity_scalar_mix_queries}")
    print(f"    joinedload         : {stats.true_orm_joinedload_queries}")
    print(f"    selectinload       : {stats.true_orm_selectinload_queries}")
    print(f"    relationship touch : {stats.true_orm_relationship_touch_queries}")
    print(f"    self alias         : {stats.true_orm_self_alias_queries}")
    print(f"    set query          : {stats.true_orm_setop_queries}")
    print(f"    scalar subquery    : {stats.true_orm_scalar_subquery_queries}")
    print(f"    window expr        : {stats.true_orm_window_queries}")
    print(f"    derived table      : {stats.true_orm_derived_table_queries}")
    print(f"    IN limit wrap      : {stats.true_orm_limit_subquery_wrap_queries}")
    print("  Fault Smoke:")
    print(f"    attempts     : {stats.fault_smoke_attempts}")
    print(f"    detected     : {stats.fault_smoke_detected}")
    print(f"    missed       : {stats.fault_smoke_missed}")
    print(f"    boosted rows : {stats.row_budget_boost_queries}")
    print(f"  true_orm path raw : {_format_true_orm_coverage_summary(get_true_orm_coverage_snapshot())}")
    print("=" * 60)

    if not stats.bug_reports:
        print("No actionable bugs found.")
        return

    print(f"\nFound {len(stats.bug_reports)} issue(s)")
    print(f"  repro scripts : {bug_dir}/")
    print("  bug details   : logs_bug/<bug_ts>.log\n")
    for i, report in enumerate(stats.bug_reports, start=1):
        print("-" * 50)
        print(f"Issue #{i}  (schema={report.schema_id + 1}, query={report.query_id + 1})")
        if report.error:
            print("  type  : execution error")
            print(f"  error : {report.error}")
        else:
            _, label, reason = _get_report_category(report)
            print(f"  type  : {label}")
            print(f"  reason: {reason}")
            print(f"  ref rows      : {report.ref_rows[:3]}{'...' if len(report.ref_rows) > 3 else ''}")
            print(f"  sql rows      : {report.sql_rows[:3]}{'...' if len(report.sql_rows) > 3 else ''}")
            print(f"  true_orm rows : {report.true_orm_rows[:3]}{'...' if len(report.true_orm_rows) > 3 else ''}")
        print(f"  IR:\n{report.ir_str}")
        print(f"  repro: {os.path.join(bug_dir, f'bug_{i:03d}.py')}")


def _choose_stress_mode(query_seed: int, stats: Optional[RunStats] = None) -> str:
    """Use deterministic weighted sampling and boost under-covered structures."""
    rng = random.Random(query_seed ^ 0x5F3759DF)
    weights = {
        "balanced": 1.8,
        "join_heavy": 1.0,
        "relationship_heavy": 0.95,
        "relationship_orderby_heavy": 0.8,
        "entity_heavy": 0.9,
        "entity_dedup_heavy": 0.8,
        "distinct_entity_heavy": 0.72,
        "limit_joined_entity_heavy": 0.72,
        "self_join_heavy": 0.8,
        "groupby_heavy": 1.0,
        "duplicate_column_heavy": 0.9,
        "null_heavy": 0.9,
        "orderby_heavy": 1.0,
        "distinct_heavy": 0.9,
        "subquery_heavy": 1.0,
        "derived_heavy": 0.72,
        "window_heavy": 0.72,
        "setop_heavy": 0.8,
        "loader_heavy": 0.75,
        "loader_strategy_heavy": 0.72,
        "orm_combo_heavy": 0.75,
        "combo_heavy": 1.0,
    }

    if stats is not None and stats.total_queries > 0:
        total = stats.total_queries

        def deficit(actual: int, target_ratio: float) -> float:
            target_count = max(1.0, total * target_ratio)
            return max(0.0, (target_count - actual) / target_count)

        join_gap = deficit(stats.join_queries, 0.5)
        multi_join_gap = deficit(stats.multi_join_queries, 0.12)
        left_join_gap = deficit(stats.left_join_queries, 0.08)
        groupby_gap = deficit(stats.groupby_queries, 0.45)
        having_gap = deficit(stats.having_queries, 0.3)
        distinct_gap = deficit(stats.distinct_queries, 0.12)
        orderby_gap = deficit(stats.orderby_queries, 0.18)
        orderby_agg_gap = deficit(stats.orderby_agg_queries, 0.06)
        duplicate_gap = deficit(stats.duplicate_proj_queries, 0.12)
        null_gap = deficit(stats.null_predicate_queries, 0.08)
        subquery_gap = deficit(stats.subquery_queries, 0.12)
        exists_gap = deficit(stats.exists_subquery_queries, 0.05)
        in_subquery_gap = deficit(stats.in_subquery_queries, 0.06)
        distinct_order_limit_gap = deficit(stats.distinct_order_limit_queries, 0.05)
        self_join_gap = deficit(stats.self_join_queries, 0.04)
        set_query_gap = deficit(stats.set_query_queries, 0.05)
        entity_gap = deficit(stats.entity_projection_queries, 0.08)
        entity_scalar_gap = deficit(stats.entity_scalar_mix_queries, 0.05)
        left_combo_gap = deficit(stats.left_join_null_queries, 0.025)
        right_proj_gap = deficit(stats.left_join_right_proj_queries, 0.04)
        duplicate_row_gap = deficit(stats.duplicate_row_results, 0.14)
        null_row_gap = deficit(stats.null_containing_results, 0.2)
        entity_duplicate_gap = deficit(stats.entity_duplicate_results, 0.06)
        left_null_ext_gap = deficit(stats.left_join_null_extension_results, 0.04)
        agg_null_gap = deficit(stats.aggregation_null_results, 0.05)
        many_row_gap = deficit(stats.multi_row_results, 0.28)
        orm_cov = get_true_orm_coverage_snapshot()
        relationship_join_gap = deficit(orm_cov.get("relationship_join_used", 0), 0.08)
        loader_gap = deficit(
            orm_cov.get("joinedload_used", 0) + orm_cov.get("selectinload_used", 0),
            0.08,
        )
        scalar_subq_gap = deficit(orm_cov.get("scalar_subquery_used", 0), 0.05)
        derived_gap = deficit(orm_cov.get("derived_table_used", 0), 0.04)
        window_gap = deficit(orm_cov.get("window_expr_used", 0), 0.03)
        explicit_fallback_gap = deficit(orm_cov.get("relationship_join_fallback", 0), 0.04)

        weights["join_heavy"] += 2.6 * join_gap + 2.2 * multi_join_gap + 1.6 * right_proj_gap
        weights["relationship_heavy"] += 2.4 * join_gap + 2.2 * right_proj_gap + 1.8 * left_join_gap
        weights["relationship_orderby_heavy"] += 2.4 * relationship_join_gap + 2.0 * orderby_gap + 1.8 * right_proj_gap
        weights["entity_heavy"] += 2.8 * entity_gap + 2.2 * entity_scalar_gap + 1.2 * join_gap
        weights["entity_dedup_heavy"] += 2.6 * entity_duplicate_gap + 2.1 * duplicate_row_gap + 1.4 * many_row_gap
        weights["distinct_entity_heavy"] += 2.2 * entity_gap + 2.0 * distinct_gap + 1.8 * distinct_order_limit_gap
        weights["limit_joined_entity_heavy"] += 2.2 * entity_gap + 2.0 * many_row_gap + 1.8 * distinct_order_limit_gap
        weights["self_join_heavy"] += 3.2 * self_join_gap + 1.2 * orderby_gap
        weights["groupby_heavy"] += 2.8 * groupby_gap + 1.8 * having_gap + 1.8 * orderby_agg_gap
        weights["duplicate_column_heavy"] += 2.4 * duplicate_gap + 1.6 * distinct_gap
        weights["null_heavy"] += 2.6 * null_gap + 2.2 * left_join_gap + 1.8 * left_combo_gap + 2.0 * null_row_gap + 1.8 * left_null_ext_gap
        weights["orderby_heavy"] += 3.0 * orderby_gap + 1.4 * right_proj_gap + 1.2 * orderby_agg_gap
        weights["distinct_heavy"] += 2.8 * distinct_gap + 1.6 * duplicate_gap
        weights["subquery_heavy"] += 3.0 * subquery_gap + 1.8 * exists_gap + 1.8 * in_subquery_gap + 1.6 * scalar_subq_gap
        weights["derived_heavy"] += 2.8 * derived_gap + 1.6 * subquery_gap + 1.4 * distinct_order_limit_gap
        weights["window_heavy"] += 3.0 * window_gap + 1.4 * orderby_gap + 1.2 * groupby_gap
        weights["setop_heavy"] += 3.0 * set_query_gap + 1.4 * subquery_gap
        weights["loader_heavy"] += 2.4 * entity_gap + 2.0 * distinct_order_limit_gap + 1.2 * join_gap
        weights["loader_strategy_heavy"] += 2.4 * loader_gap + 2.0 * entity_gap + 1.5 * entity_duplicate_gap
        weights["orm_combo_heavy"] += 2.6 * entity_gap + 2.4 * set_query_gap + 2.0 * self_join_gap
        weights["combo_heavy"] += (
            2.6 * multi_join_gap
            + 2.4 * distinct_order_limit_gap
            + 1.8 * subquery_gap
            + 1.8 * left_combo_gap
            + 1.6 * orderby_agg_gap
            + 1.4 * window_gap
        )
        weights["relationship_heavy"] += 1.6 * explicit_fallback_gap
        weights["groupby_heavy"] += 1.8 * agg_null_gap

        if total < 50:
            weights["balanced"] *= 0.7
            weights["join_heavy"] += 0.6
            weights["relationship_heavy"] += 0.4
            weights["relationship_orderby_heavy"] += 0.35
            weights["entity_heavy"] += 0.5
            weights["entity_dedup_heavy"] += 0.35
            weights["distinct_entity_heavy"] += 0.25
            weights["setop_heavy"] += 0.3
            weights["groupby_heavy"] += 0.4
            weights["orderby_heavy"] += 0.4
            weights["subquery_heavy"] += 0.5
            weights["derived_heavy"] += 0.25
            weights["window_heavy"] += 0.25
            weights["loader_heavy"] += 0.3
            weights["loader_strategy_heavy"] += 0.25
            weights["limit_joined_entity_heavy"] += 0.25
            weights["combo_heavy"] += 0.5

    total_weight = sum(weights.values())
    roll = rng.random() * total_weight
    upto = 0.0
    for mode, weight in weights.items():
        upto += weight
        if roll <= upto:
            return mode
    return "balanced"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="RetORM differential testing framework")
    parser.add_argument("--schemas",  type=int,  default=config.NUM_SCHEMAS)
    parser.add_argument("--queries",  type=int,  default=config.QUERIES_PER_SCHEMA)
    parser.add_argument("--tables",   type=int,  default=2)
    parser.add_argument("--cols",     type=int,  default=3)
    parser.add_argument("--rows",     type=int,  default=config.RANDOM_ROWS)
    parser.add_argument("--seed",     type=int,  default=None)
    parser.add_argument("--fault-smoke-rate", type=float, default=config.TRUE_ORM_FAULT_SMOKE_RATE)
    parser.add_argument("--strict-compare", action="store_true")
    parser.add_argument("--true-orm-fault", type=str, default=config.TRUE_ORM_FAULT_INJECTION)
    parser.add_argument("--no-z3",   action="store_true")
    parser.add_argument("--verbose",  action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config.TRUE_ORM_FAULT_SMOKE_RATE = max(0.0, float(args.fault_smoke_rate))
    config.TRUE_ORM_FAULT_INJECTION = args.true_orm_fault or "off"
    run(
        num_schemas        = args.schemas,
        queries_per_schema = args.queries,
        num_tables         = args.tables,
        cols_per_table     = args.cols,
        rows_per_table     = args.rows,
        use_z3             = not args.no_z3,
        seed               = args.seed,
        verbose            = args.verbose,
        strict_compare     = args.strict_compare,
    )


