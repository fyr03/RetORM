"""
runner.py

RetORM 主入口，串联整个自动化测试流程。

日志：
  logs_detail/YYMMDD_HHMMSS.log  详细日志，和 --verbose 终端输出内容一致
  logs/YYMMDD_HHMMSS.log         运行日志，每 10 秒输出一次宏观统计
  logs_bug/YYMMDD_HHMMSS.log     bug 详情日志（IR/SQL/ORM API/program code）

bug 输出：
  bugs/YYMMDD_HHMMSS/bug_N.py    可直接运行的复现脚本

用法：
  python runner.py
  python runner.py --schemas 3 --queries 10 --seed 42 --no-z3 --verbose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import random
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

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
    reset_model_cache,
    supports_true_orm,
    UnsupportedTrueORM,
)
from comparator.compare         import compare_all, compare_two_paths, print_report
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
# 日志系统初始化
# ---------------------------------------------------------------------------

def _setup_logging(run_ts: str):
    """
    建立两个 logger：
      detail_logger : 详细日志，写入 logs_detail/<ts>.log
      run_logger    : 运行日志，写入 logs/<ts>.log

    bug 详情日志在每次发现 bug 时单独创建，文件名用 bug 发现时刻的时间戳。
    终端输出仍然用 print()，保持原有交互体验不变。
    """
    os.makedirs("logs_detail", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("logs_bug",    exist_ok=True)

    # 详细日志：无时间戳，只输出内容，保持可读性
    detail_fmt = logging.Formatter("%(message)s")
    detail_logger = logging.getLogger("retorm.detail")
    detail_logger.setLevel(logging.DEBUG)
    dh = logging.FileHandler(f"logs_detail/{run_ts}.log", encoding="utf-8")
    dh.setFormatter(detail_fmt)
    detail_logger.addHandler(dh)
    detail_logger.propagate = False

    # 运行日志：带时间戳
    run_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    run_logger = logging.getLogger("retorm.run")
    run_logger.setLevel(logging.INFO)
    rh = logging.FileHandler(f"logs/{run_ts}.log", encoding="utf-8")
    rh.setFormatter(run_fmt)
    run_logger.addHandler(rh)
    run_logger.propagate = False

    return detail_logger, run_logger


# ---------------------------------------------------------------------------
# 每 10 秒输出一次统计的后台线程
# ---------------------------------------------------------------------------

class _StatsPrinter(threading.Thread):
    """后台线程，每 interval 秒向 run_logger 写一次宏观统计。"""

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
        self._log()   # 停止时再写一次最终状态

    def _log(self):
        elapsed = int(time.time() - self.start_time)
        s = self.stats
        total = s.total_queries

        # 比率计算（避免除零）
        pass_rate   = f"{s.passed / total * 100:.1f}%" if total else "N/A"
        empty_rate  = f"{s.empty_results / total * 100:.1f}%" if total else "N/A"
        error_rate  = f"{s.errors / total * 100:.1f}%" if total else "N/A"
        bug_total   = s.sql_bugs + s.sql_true_orm_divergences
        bug_rate    = f"{bug_total / total * 100:.1f}%" if total else "N/A"

        # 速度：queries per minute
        qpm = f"{total / elapsed * 60:.1f}" if elapsed > 0 else "N/A"

        lines = [
            f"[运行统计]  耗时={elapsed}s  速度={qpm} q/min , 查询总数  : {total}",
            f"  通过      : {s.passed}  ({pass_rate}),  空结果    : {s.empty_results}  ({empty_rate}),  执行错误  : {s.errors}  ({error_rate})",
            f"  SQL bug   : {s.sql_bugs},  SQL vs true ORM diff  : {s.sql_true_orm_divergences},  bug 合计  : {bug_total}  ({bug_rate})",
            f"  结构覆盖  : 单表={s.single_table_queries}, Join={s.join_queries}, LEFT JOIN={s.left_join_queries}, Filter={s.filter_queries}, GroupBy={s.groupby_queries}, Having={s.having_queries}, 重复投影={s.duplicate_proj_queries}, NULL谓词={s.null_predicate_queries}",
        ]
        lines = _format_run_stats_lines(
            s, elapsed, qpm, pass_rate, empty_rate, error_rate, bug_rate
        )
        for line in lines:
            self.run_logger.info(line)


# ---------------------------------------------------------------------------
# Bug 记录 & 统计
# ---------------------------------------------------------------------------

@dataclass
class BugReport:
    schema_id:    int
    query_id:     int
    schema:       object          # Schema 对象，用于生成复现代码
    ir:           object          # 原始 IR 对象，用于稳定复现
    ir_str:       str
    schema_seed:  int
    query_seed:   int
    table_data:   dict            # 原始插入数据，避免复现时重新随机生成
    rows_per_table: int
    use_z3:       bool
    z3_timeout:   int
    ref_vs_sql:   object          # CompareResult
    ref_vs_true_orm: object       # CompareResult
    ref_rows:     list
    sql_rows:     list
    true_orm_rows: list
    sql_vs_true_orm: object = None   # CompareResult
    sql_text:     str = ""        # 生成的 SQL 字符串
    error:        Optional[str] = None


@dataclass
class RunStats:
    total_queries: int = 0
    passed:        int = 0
    sql_bugs:      int = 0
    sql_true_orm_divergences: int = 0
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
    bug_reports:   List[BugReport] = field(default_factory=list)


def _report_has_actionable_bug(report: BugReport) -> bool:
    """
    A report should only be persisted when it represents a real SQL vs true ORM
    finding or an execution failure. Ref-path mismatches are diagnostic only.
    """
    if report.error:
        return True
    if report.sql_vs_true_orm is None:
        return False
    return not report.sql_vs_true_orm.match


def _get_report_category(report: BugReport) -> tuple[str, str, str]:
    if report.error:
        return "execution_error", "Execution Error", report.error

    ref_vs_sql = report.ref_vs_sql
    ref_vs_true_orm = report.ref_vs_true_orm
    sql_vs_true_orm = report.sql_vs_true_orm
    sql_mismatch = ref_vs_sql is not None and not ref_vs_sql.match
    true_orm_mismatch = ref_vs_true_orm is not None and not ref_vs_true_orm.match
    sql_true_orm_mismatch = sql_vs_true_orm is not None and not sql_vs_true_orm.match

    if sql_vs_true_orm is None:
        reason_parts = []
        if sql_mismatch:
            reason_parts.append(f"ref_vs_sql: {ref_vs_sql.reason}")
        if true_orm_mismatch:
            reason_parts.append(f"ref_vs_true_orm: {ref_vs_true_orm.reason}")
        return "ref_path_anomaly", "Ref Path Anomaly", " | ".join(reason_parts)

    if not sql_true_orm_mismatch:
        if sql_mismatch or true_orm_mismatch:
            reason_parts = []
            if sql_mismatch:
                reason_parts.append(f"ref_vs_sql: {ref_vs_sql.reason}")
            if true_orm_mismatch:
                reason_parts.append(f"ref_vs_true_orm: {ref_vs_true_orm.reason}")
            return "ref_path_anomaly", "Ref Path Anomaly", " | ".join(reason_parts)
        return "consistent", "Consistent", ""

    reason_parts = [f"sql_vs_true_orm: {sql_vs_true_orm.reason}"]
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

    if report.error:
        return "鎵ц寮傚父", report.error

    ref_vs_sql = report.ref_vs_sql
    ref_vs_orm = report.ref_vs_orm
    sql_mismatch = ref_vs_sql is not None and not ref_vs_sql.match
    orm_mismatch = ref_vs_orm is not None and not ref_vs_orm.match

    if sql_mismatch and orm_mismatch:
        reason = (
            f"ref_vs_sql: {ref_vs_sql.reason} | "
            f"ref_vs_orm: {ref_vs_orm.reason}"
        )
        return "ref 璺緞鍙兘鏈?bug锛坮ef vs sql 鍜?ref vs orm 鍧囦笉涓€鑷达級", reason
    if sql_mismatch:
        return "SQL 缈昏瘧鍣?bug", ref_vs_sql.reason
    if orm_mismatch:
        return "legacy core-path diff", ref_vs_orm.reason
    return "鏈煡锛堜笁璺潎涓€鑷达紝涓嶅簲鍑虹幇锛?", ""


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
# Bug 报告文件生成
# ---------------------------------------------------------------------------

def _write_bug_detail(report: BugReport) -> str:
    """
    把 bug 的完整技术细节写入 logs_bug/<bug_ts>.log。
    文件名用 bug 发现时刻的时间戳，每个 bug 独立一个文件。

    包含：
      1. IR 树（文字表示）
      2. 生成的 Raw SQL
      3. ORM API 等价调用说明
      4. Program code（python_ref 的执行逻辑说明）
      5. 三路结果对比

    返回写出的文件路径。
    """
    bug_ts = datetime.now().strftime("%y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
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

    sep = "=" * 70

    # ── 确定 bug 类型 ──
    _, bug_type, reason = _get_report_category(report)

    lines = [
        sep,
        f"BUG #{report.schema_id + 1}-{report.query_id + 1}",
        f"类型       : {bug_type}",
        f"原因       : {reason}",
        f"Schema seed: {report.schema_seed}",
        f"Query  seed: {report.query_seed}",
        f"rows/table : {report.rows_per_table}",
        f"use_z3     : {report.use_z3}",
        f"z3_timeout : {report.z3_timeout}s",
        sep,
        "",
        "── Schema（建表 SQL）──",
    ]
    for sql in generate_create_sqls(report.schema):
        lines.append(sql)
    lines.append("")

    lines += [
        "── IR 树 ──",
        report.ir_str,
        "",
        "── 原始插入数据 ──",
    ]
    for table in report.schema.tables:
        rows = report.table_data.get(table.name, [])
        lines.append(f"  {table.name} ({len(rows)} 行): {rows}")
    lines.append("")

    lines += [
        "── Raw SQL（路径二）──",
    ]

    # Raw SQL
    if report.sql_text:
        sql_str = report.sql_text
    else:
        try:
            sql_str = sql_translate(report.ir)
        except Exception as e:
            sql_str = f"（生成失败: {e}）"

    lines += [
        sql_str,
        "",
        "── True ORM API（路径三：SQLAlchemy ORM）──",
        "  通过 translators/sqlalchemy_true_orm.py 的 execute(ir) 执行，",
        "  SQLAlchemy 使用 mapped classes + Session 来构建查询，经过以下步骤：",
        "    1. 根据 schema 动态生成 declarative models 和 relationship()",
        "    2. 将 IR 翻译成 ORM select / join / filter / group by / subquery",
        "    3. session.execute(stmt) 提交给 MySQL 执行，并将实体结果展开回行",
        "  等价于以下 SQLAlchemy ORM 调用链（伪代码）：",
        _gen_orm_pseudocode(report),
        "",
        "── Program Code（路径一：python_ref）──",
        "  通过 translators/python_ref.py 的 execute(ir) 执行：",
        "    1. Scan   : SELECT * FROM table，列名加表别名前缀",
        "    2. Join   : 嵌套循环 INNER JOIN",
        "    3. Filter : Python 列表推导过滤",
        "    4. GroupBy: defaultdict 分桶 + 逐组聚合",
        "    5. Having : 分组后再次过滤",
        "    6. Project: 只保留指定列",
        "",
        "── 三路执行结果对比 ──",
        f"  ref（程序逻辑）{len(report.ref_rows)} 行: {report.ref_rows}",
        f"  sql（Raw SQL） {len(report.sql_rows)} 行: {report.sql_rows}",
        f"  true_orm（ORM API） {len(report.true_orm_rows)} 行: {report.true_orm_rows}",
        "",
    ]

    if report.ref_vs_sql is not None and not report.ref_vs_sql.match:
        lines.append(f"  ref vs sql 不一致: {report.ref_vs_sql.reason}")
    if report.ref_vs_true_orm is not None and not report.ref_vs_true_orm.match:
        lines.append(f"  ref vs true_orm 不一致: {report.ref_vs_true_orm.reason}")
    if report.sql_vs_true_orm is not None and not report.sql_vs_true_orm.match:
        lines.append(f"  sql vs true_orm 不一致: {report.sql_vs_true_orm.reason}")

    lines.append(sep)

    bug_logger.info("\n".join(lines))
    # 关闭这个 logger 的 handler，避免重复写入
    bh.close()
    bug_logger.removeHandler(bh)
    return fpath_detail


def _gen_orm_pseudocode(report: BugReport) -> str:
    """根据 IR 生成 SQLAlchemy ORM 的伪代码描述。"""
    from ir.nodes import Scan, Filter, Join, GroupBy, Having, Project

    # 直接把 IR 结构转成伪代码注释，简单可靠
    lines = ["    # IR → SQLAlchemy ORM 伪代码："]
    for line in report.ir_str.split("\n"):
        lines.append("    # " + line)
    return "\n".join(lines)


def _write_bug_file(report: BugReport, bug_dir: str, bug_idx: int) -> str:
    """
    把一个 bug 写成可直接运行的 Python 复现脚本。
    返回文件路径。
    """
    if not _report_has_actionable_bug(report):
        raise ValueError("attempted to write a non-actionable bug repro script")
    os.makedirs(bug_dir, exist_ok=True)
    fpath = os.path.join(bug_dir, f"bug_{bug_idx:03d}.py")

    # ── 确定 bug 类型 ──
    _, bug_type, reason = _get_report_category(report)

    # ── 生成建表 SQL ──
    from generator.schema_gen import generate_create_sqls, generate_drop_sqls
    create_sqls = generate_create_sqls(report.schema)
    drop_order  = generate_drop_sqls(report.schema)

    # ── 直接保存原始插入数据和原始 IR，避免复现依赖后续生成器行为 ──
    insert_code = _gen_insert_code(report.schema, report.table_data)
    ir_code = _gen_ir_code(report.ir)

    # ── 复现脚本 ──
    repro_script = textwrap.dedent(f"""\
    #!/usr/bin/env python3
    \"\"\"
    RetORM Bug 复现脚本
    ==================
    Bug 类型   : {bug_type}
    原因       : {reason}
    Schema ID  : {report.schema_id + 1}
    Query  ID  : {report.query_id + 1}
    Schema seed: {report.schema_seed}
    Query  seed: {report.query_seed}
    rows/table : {report.rows_per_table}
    use_z3     : {report.use_z3}
    z3_timeout : {report.z3_timeout}s

    运行方式：
        conda activate retorm
        python {os.path.basename(fpath)}
    \"\"\"

    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from db.connector import (
        init_database, create_tables, drop_tables, insert_rows,
    )
    from translators.python_ref     import execute as ref_execute
    from translators.sql            import execute as sql_execute, translate as sql_translate
    from translators.sqlalchemy_true_orm import execute as true_orm_execute, reset_model_cache
    from comparator.compare         import compare_all, print_report
    from ir.nodes import *
    from runner import _query_requires_ordered_compare

    # ── 1. 建表 ────────────────────────────────────────────────────────────
    init_database()
    drop_tables({drop_order!r})
    create_tables({create_sqls!r})
    reset_model_cache()

    # ── 2. 插入数据 ─────────────────────────────────────────────────────────
    {insert_code}

    # ── 3. 构造 IR ──────────────────────────────────────────────────────────
    # 直接使用 bug 发现时捕获的 IR，避免后续生成器变化导致复现失真
    ir = {ir_code}

    print("IR 树：")
    from ir.nodes import pretty_print
    print(pretty_print(ir))

    # ── 4. 生成 SQL（供参考）──────────────────────────────────────────────
    print("\\n生成的 SQL：")
    print(sql_translate(ir))

    # ── 5. 三路执行 ────────────────────────────────────────────────────────
    ref_rows = ref_execute(ir)
    sql_rows = sql_execute(ir)
    true_orm_rows = true_orm_execute(ir, schema)

    print(f"\\nref 结果 ({{len(ref_rows)}} 行): {{ref_rows}}")
    print(f"sql 结果 ({{len(sql_rows)}} 行): {{sql_rows}}")
    print(f"true_orm 结果 ({{len(true_orm_rows)}} 行): {{true_orm_rows}}")

    # ── 6. 比较 ────────────────────────────────────────────────────────────
    ordered = _query_requires_ordered_compare(ir)
    ref_vs_sql, ref_vs_true_orm = compare_all(
        ref_rows, sql_rows, true_orm_rows, ordered=ordered
    )
    print_report(ref_vs_sql, ref_vs_true_orm)

    # ── 7. 清理 ────────────────────────────────────────────────────────────
    drop_tables({drop_order!r})
    print("\\n复现完成。")
    """)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(repro_script)

    return fpath


def _gen_insert_code(schema, table_data: dict) -> str:
    """生成插入真实原始数据的代码片段，避免复现时重新随机生成。"""
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


def _gen_ir_code(node) -> str:
    """把当前 IR 对象序列化为可直接执行的 Python 构造代码。"""
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
    raise TypeError(f"不支持序列化的 IR 节点: {type(node)}")


# ---------------------------------------------------------------------------
# 主流程
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
) -> RunStats:

    run_ts  = datetime.now().strftime("%y%m%d_%H%M%S")
    bug_dir = os.path.join("bugs", run_ts)

    detail_logger, run_logger = _setup_logging(run_ts)

    def dlog(msg: str):
        """同时写详细日志文件，终端不重复打印（终端由 print 负责）。"""
        detail_logger.info(msg)

    # 启动后台统计线程
    stats   = RunStats()
    printer = _StatsPrinter(stats, run_logger, interval=10)
    printer.start()

    # ── 启动信息 ──────────────────────────────────────────────────────────
    header = (
        f"{'=' * 60}\n"
        f"RetORM 差分测试启动  [{run_ts}]\n"
        f"  schemas={num_schemas}, queries/schema={queries_per_schema}\n"
        f"  tables={num_tables}, cols={cols_per_table}, base_rows={rows_per_table}\n"
        f"  use_z3={use_z3}, seed={seed}\n"
        f"  detail_log=logs_detail/{run_ts}.log\n"
        f"  run_log=logs/{run_ts}.log\n"
        f"{'=' * 60}"
    )
    print(header)
    dlog(header)
    run_logger.info("=" * 50)
    run_logger.info(f"RetORM 差分测试启动  [{run_ts}]")
    run_logger.info(f"  schemas={num_schemas}  queries/schema={queries_per_schema}")
    run_logger.info(f"  tables={num_tables}  cols/table={cols_per_table}")
    run_logger.info(f"  use_z3={use_z3}  seed={seed}")
    run_logger.info(
        "  row budget  : "
        f"base={rows_per_table}  extra_random={config.EXTRA_RANDOM_ROWS}  "
        f"edge={config.EDGE_ROWS}  adversarial={config.ADVERSARIAL_ROWS}"
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
            sep = f"\n{'─' * 50}\nSchema {schema_id + 1}/{num_schemas}  (seed={schema_seed})"
            print(sep)
            dlog(sep)
            run_logger.info(
                f"[Schema {schema_id + 1}/{num_schemas}]  开始  "
                f"seed={schema_seed}  tables={num_tables}"
            )

            # ── 生成 Schema ──────────────────────────────────────────────
            schema = generate_schema(
                num_tables=num_tables,
                cols_per_table=cols_per_table,
                fk_prob=0.6,
                seed=schema_seed,
            )
            if verbose:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    print_schema(schema)
                schema_desc = buf.getvalue()
                print(schema_desc, end="")
                dlog(schema_desc)

            # ── 建表 ─────────────────────────────────────────────────────
            try:
                drop_tables(generate_drop_sqls(schema))
                create_tables(generate_create_sqls(schema))
                reset_model_cache()
            except Exception as e:
                msg = f"[runner] 建表失败，跳过此 schema: {e}"
                print(msg); dlog(msg)
                continue

            # ── 每条 IR ──────────────────────────────────────────────────
            for query_id in range(queries_per_schema):
                query_seed = schema_seed + query_id + 1
                stats.total_queries += 1
                table_data = {}
                stress_mode = _choose_stress_mode(query_seed, stats)

                prefix = (
                    f"\n  Query {query_id + 1}/{queries_per_schema}  "
                    f"(seed={query_seed}, mode={stress_mode})"
                )
                print(prefix, end="  ")
                # 3. 生成 IR
                try:
                    ir, ctx = generate_ir(schema, stress_mode=stress_mode, seed=query_seed)
                except Exception as e:
                    msg = f"[IR生成失败] {e}"
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
                # 一次性写 query 头 + IR，不分两次
                dlog(f"{prefix}\nIR:\n{ir_str}")

                # 4. 生成数据
                try:
                    _truncate_schema(schema)
                    table_data = generate_and_insert(
                        schema, ir,
                        rows_per_table=rows_per_table,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        stress_mode=stress_mode,
                        seed=query_seed,
                    )
                except Exception as e:
                    msg = f"[数据生成失败] {e}"
                    print(msg); dlog(msg)
                    stats.errors += 1
                    continue

                # 5. 三路执行
                sql_text = ""
                try:
                    ref_rows = ref_execute(ir) if config.ENABLE_REF_PATH else []
                    sql_rows = sql_execute(ir)
                    with _temporary_true_orm_runtime(stress_mode, ctx):
                        true_orm_rows = true_orm_execute(ir, schema) if config.ENABLE_TRUE_ORM_PATH else []
                    sql_text = sql_translate(ir)

                    if verbose:
                        exec_info = (
                            f"  ref: {ref_rows}\n"
                            f"  sql: {sql_rows}\n"
                            f"  true_orm: {true_orm_rows}\n"
                            f"  SQL: {sql_text}"
                        )
                        print(exec_info)
                        # exec_info 在后面和结论合并写，这里不单独 dlog
                except Exception as e:
                    msg = f"[执行失败] {e}"
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
                        rows_per_table=rows_per_table,
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
                    msg2 = f"  → 复现脚本: {fpath}  bug详情: {fpath_detail}"
                    print(msg2); dlog(msg2)
                    continue

                # 6. 比较
                try:
                    ordered = _query_requires_ordered_compare(ir)
                    if config.ENABLE_REF_PATH:
                        ref_vs_sql, ref_vs_true_orm = compare_all(
                            ref_rows, sql_rows, true_orm_rows, ordered=ordered
                        )
                    else:
                        ref_vs_sql, ref_vs_true_orm = None, None
                    sql_vs_true_orm = compare_two_paths(
                        sql_rows, true_orm_rows, "sql", "true_orm", ordered=ordered
                    )
                except Exception as e:
                    msg = f"[比较失败] {e}"
                    print(msg); dlog(msg)
                    stats.errors += 1
                    continue

                # 7. 统计 & 日志
                all_empty = (len(ref_rows) == 0 and
                             len(sql_rows) == 0 and
                             len(true_orm_rows) == 0)
                if all_empty:
                    print("(空结果)", end="")
                    stats.empty_results += 1

                if sql_vs_true_orm.match:
                    print("✓")
                    # 通过时一行总结，verbose 时附上三路结果
                    if verbose:
                        dlog(f"  ref({len(ref_rows)}行) sql({len(sql_rows)}行) "
                             f"true_orm({len(true_orm_rows)}行)  → ✓ 三路一致"
                             + ("  [空结果]" if all_empty else ""))
                    else:
                        dlog(f"  → ✓ 三路一致"
                             + ("  [空结果]" if all_empty else ""))
                    stats.passed += 1
                else:
                    report = BugReport(
                        schema_id=schema_id, query_id=query_id,
                        schema=schema, ir=ir, ir_str=ir_str,
                        schema_seed=schema_seed, query_seed=query_seed,
                        table_data=table_data,
                        rows_per_table=rows_per_table,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        ref_vs_sql=ref_vs_sql, ref_vs_true_orm=ref_vs_true_orm,
                        ref_rows=ref_rows, sql_rows=sql_rows, true_orm_rows=true_orm_rows,
                        sql_vs_true_orm=sql_vs_true_orm,
                        sql_text=sql_text,
                    )
                    bug_category, tag, reason = _get_report_category(report)
                    if bug_category == "sql_true_orm_divergence":
                        stats.sql_true_orm_divergences += 1
                        tag = "SQL vs true_orm divergence"
                    else:
                        stats.ref_path_anomalies += 1
                        tag = "△ ref path anomaly"

                    print(tag)
                    dlog(f"  ref({len(ref_rows)}行) sql({len(sql_rows)}行) "
                         f"true_orm({len(true_orm_rows)}行)  → {tag}")

                    # 写详细比较到详细日志
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        print_report(ref_vs_sql, ref_vs_true_orm, ir_str)
                    report_str = buf.getvalue()
                    if verbose:
                        print(report_str)
                    dlog(report_str)

                    bug_idx = _persist_bug_report(
                        report, stats, bug_dir, bug_idx, dlog
                    )
                    continue
                    stats.bug_reports.append(report)
                    fpath = _write_bug_file(report, bug_dir, bug_idx)
                    fpath_detail = _write_bug_detail(report)
                    msg2 = f"  → 复现脚本: {fpath}  bug详情: {fpath_detail}"
                    print(msg2); dlog(msg2)

            # ── 清理 + schema 小结 ───────────────────────────────────────
            try:
                drop_tables(generate_drop_sqls(schema))
            except Exception as e:
                print(f"[runner] 清理失败: {e}")

            # schema 级别小结写入运行日志
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
                f"[Schema {schema_id + 1}/{num_schemas}]  结束  "
                f"查询={schema_total}  bug={schema_bugs}  错误={schema_errors}"
            )

    finally:
        printer.stop()

    # ── 最终报告 ──────────────────────────────────────────────────────────
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
    run_logger.info(f"RetORM 测试结束  [{run_ts}]")
    run_logger.info(f"  查询总数  : {total}")
    run_logger.info(f"  通过      : {s.passed}  ({pass_rate})")
    run_logger.info(f"  空结果    : {s.empty_results}  ({empty_rate})")
    run_logger.info(f"  执行错误  : {s.errors}")
    run_logger.info(f"  SQL bug   : {s.sql_bugs}")
    run_logger.info(f"  SQL vs true ORM diff : {s.sql_true_orm_divergences}")
    run_logger.info(f"  ref anomaly  : {s.ref_path_anomalies}")
    run_logger.info(f"  true_orm unsupported : {s.true_orm_unsupported}")
    run_logger.info(f"  bug 合计  : {s.sql_bugs + s.sql_true_orm_divergences}")
    run_logger.info(
        "  结构覆盖  : "
        f"单表={s.single_table_queries}  Join={s.join_queries}  LEFT JOIN={s.left_join_queries}  "
        f"SelfJoin={s.self_join_queries}  SetQuery={s.set_query_queries}  "
        f"EntityProj={s.entity_projection_queries}  Entity+Scalar={s.entity_scalar_mix_queries}  "
        f"Filter={s.filter_queries}  GroupBy={s.groupby_queries}  "
        f"Having={s.having_queries}  重复投影={s.duplicate_proj_queries}  NULL谓词={s.null_predicate_queries}"
    )
    if s.bug_reports:
        run_logger.info(f"  复现脚本  : {bug_dir}/")
        run_logger.info(f"  bug详情   : logs_bug/<bug_ts>.log")
    run_logger.info("=" * 50)

    dispose_engine()
    return stats


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _truncate_schema(schema: Schema) -> None:
    for tname in generate_drop_sqls(schema):
        execute_sql(f"TRUNCATE TABLE `{tname}`;")


@contextmanager
def _temporary_true_orm_runtime(stress_mode: str, ctx) -> None:
    old_strategy = getattr(config, "TRUE_ORM_LOADER_STRATEGY", "off")
    old_touch = getattr(config, "TRUE_ORM_TOUCH_RELATIONSHIPS", False)

    strategy = old_strategy
    touch = old_touch
    has_entity_projection = bool(getattr(ctx, "projected_entity_aliases", set()))

    if has_entity_projection:
        if stress_mode == "loader_heavy":
            strategy = random.choice(["joined", "selectin"])
            touch = True
        elif stress_mode == "entity_heavy":
            strategy = "selectin"
        elif stress_mode in ("relationship_heavy", "orm_combo_heavy", "combo_heavy"):
            strategy = "joined"
            touch = random.random() < 0.5

    config.TRUE_ORM_LOADER_STRATEGY = strategy
    config.TRUE_ORM_TOUCH_RELATIONSHIPS = touch
    try:
        yield
    finally:
        config.TRUE_ORM_LOADER_STRATEGY = old_strategy
        config.TRUE_ORM_TOUCH_RELATIONSHIPS = old_touch


def _choose_stress_mode_legacy(query_seed: int) -> str:
    """基于 query_seed 可复现地选择一个压力模式。"""
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
            features["has_case_when"] = True
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
    return _collect_query_features(node)["has_orderby"]


def _format_run_stats_lines(
    stats: RunStats,
    elapsed: int,
    qpm: str,
    pass_rate: str,
    empty_rate: str,
    error_rate: str,
    bug_rate: str,
) -> List[str]:
    bug_total = stats.sql_bugs + stats.sql_true_orm_divergences
    return [
        f"[running log]  time={elapsed}s  speed={qpm} q/min , query  : {stats.total_queries}",
        f"  passed      : {stats.passed}  ({pass_rate}),  empty results   : {stats.empty_results}  ({empty_rate}),  errors  : {stats.errors}  ({error_rate})",
        f"  SQL bug   : {stats.sql_bugs},  SQL vs true ORM diff  : {stats.sql_true_orm_divergences},  ref anomaly  : {stats.ref_path_anomalies},  true_orm unsupported  : {stats.true_orm_unsupported},  bug total  : {bug_total}  ({bug_rate})",
        f"  structure coverage  : single table={stats.single_table_queries}, join={stats.join_queries}, multi join={stats.multi_join_queries}, self join={stats.self_join_queries}, left join={stats.left_join_queries}, filter={stats.filter_queries}, group by={stats.groupby_queries}, having={stats.having_queries}, distinct={stats.distinct_queries}, order by={stats.orderby_queries}, order by agg={stats.orderby_agg_queries}, set query={stats.set_query_queries}, entity proj={stats.entity_projection_queries}, entity+scalar={stats.entity_scalar_mix_queries}, duplicate projection={stats.duplicate_proj_queries}, null predicate={stats.null_predicate_queries}",
        f"  syntax coverage  : limit/offset={stats.limit_offset_queries}, IN={stats.in_list_predicate_queries}, BETWEEN={stats.between_predicate_queries}, LIKE={stats.like_predicate_queries}, arithmetic={stats.arithmetic_expr_queries}, CASE={stats.case_when_queries}, subquery={stats.subquery_queries}, EXISTS={stats.exists_subquery_queries}, IN-subquery={stats.in_subquery_queries}, distinct+order+limit={stats.distinct_order_limit_queries}",
        f"  left join combinations  : LEFT+NULL={stats.left_join_null_queries}, LEFT+GroupBy={stats.left_join_groupby_queries}, LEFT+Having={stats.left_join_having_queries}, LEFT+right projection={stats.left_join_right_proj_queries}, LEFT+right predicate={stats.left_join_right_predicate_queries}",
    ]


def _print_final_report(stats: RunStats, bug_dir: str = "bugs") -> None:
    print("\n" + "=" * 60)
    print("测试完成")
    print(f"  总查询数    : {stats.total_queries}")
    print(f"  通过        : {stats.passed}")
    print(f"  空结果      : {stats.empty_results}")
    print(f"  执行错误    : {stats.errors}")
    print(f"  SQL 翻译 bug: {stats.sql_bugs}")
    print(f"  SQL vs true ORM diff: {stats.sql_true_orm_divergences}")
    print(f"  ref anomaly : {stats.ref_path_anomalies}")
    print(f"  true_orm unsupported : {stats.true_orm_unsupported}")
    print("  Structure Coverage    :")
    print(f"    single table      : {stats.single_table_queries}")
    print(f"    Join      : {stats.join_queries}")
    print(f"    MULTI_Join : {stats.multi_join_queries}")
    print(f"    SelfJoin  : {stats.self_join_queries}")
    print(f"    LEFT JOIN : {stats.left_join_queries}")
    print(f"    Filter    : {stats.filter_queries}")
    print(f"    GroupBy   : {stats.groupby_queries}")
    print(f"    Having    : {stats.having_queries}")
    print(f"    Distinct  : {stats.distinct_queries}")
    print(f"    OrderBy   : {stats.orderby_queries}")
    print(f"    OrderByAgg: {stats.orderby_agg_queries}")
    print(f"    LimitOffset: {stats.limit_offset_queries}")
    print(f"    IN Pred   : {stats.in_list_predicate_queries}")
    print(f"    BETWEEN   : {stats.between_predicate_queries}")
    print(f"    LIKE Pred : {stats.like_predicate_queries}")
    print(f"    ArithExpr : {stats.arithmetic_expr_queries}")
    print(f"    CASE WHEN : {stats.case_when_queries}")
    print(f"    Subquery  : {stats.subquery_queries}")
    print(f"    EXISTS    : {stats.exists_subquery_queries}")
    print(f"    IN-Subq   : {stats.in_subquery_queries}")
    print(f"    Dist+Ord+Lim : {stats.distinct_order_limit_queries}")
    print(f"    SetQuery  : {stats.set_query_queries}")
    print(f"    EntityProj: {stats.entity_projection_queries}")
    print(f"    Entity+Scalar : {stats.entity_scalar_mix_queries}")
    print(f"    duplicate_proj_queries  : {stats.duplicate_proj_queries}")
    print(f"    NULL Pred   : {stats.null_predicate_queries}")
    print("  left join combinations    :")
    print(f"    LEFT+NULL : {stats.left_join_null_queries}")
    print(f"    LEFT+GB   : {stats.left_join_groupby_queries}")
    print(f"    LEFT+HAV  : {stats.left_join_having_queries}")
    print(f"    LEFT+Proj : {stats.left_join_right_proj_queries}")
    print(f"    LEFT+Pred : {stats.left_join_right_predicate_queries}")
    print("=" * 60)

    if not stats.bug_reports:
        print("未发现 bug ✓")
        return

    print(f"\n发现 {len(stats.bug_reports)} 个问题")
    print(f"  复现脚本: {bug_dir}/")
    print(f"  bug 详情: logs_bug/<bug_发现时刻>.log  （每个 bug 独立一个文件）\n")
    for i, r in enumerate(stats.bug_reports):
        print(f"{'─' * 50}")
        print(f"问题 #{i + 1}  (schema={r.schema_id + 1}, query={r.query_id + 1})")
        if r.error:
            print(f"  类型: 执行异常")
            print(f"  错误: {r.error}")
        else:
            _, label, reason = _get_report_category(r)
            print(f"  类型: {label}")
            print(f"  原因: {reason}")
            print(f"  ref ({len(r.ref_rows)}行): {r.ref_rows[:3]}"
                  f"{'...' if len(r.ref_rows) > 3 else ''}")
            print(f"  sql ({len(r.sql_rows)}行): {r.sql_rows[:3]}"
                  f"{'...' if len(r.sql_rows) > 3 else ''}")
            print(f"  true_orm ({len(r.true_orm_rows)}行): {r.true_orm_rows[:3]}"
                  f"{'...' if len(r.true_orm_rows) > 3 else ''}")
        print(f"  IR:\n{r.ir_str}")
        print(f"  复现: {os.path.join(bug_dir, f'bug_{i+1:03d}.py')}")


def _choose_stress_mode(query_seed: int, stats: Optional[RunStats] = None) -> str:
    """Use deterministic weighted sampling and boost under-covered structures."""
    rng = random.Random(query_seed ^ 0x5F3759DF)
    weights = {
        "balanced": 1.8,
        "join_heavy": 1.0,
        "relationship_heavy": 0.95,
        "entity_heavy": 0.9,
        "self_join_heavy": 0.8,
        "groupby_heavy": 1.0,
        "duplicate_column_heavy": 0.9,
        "null_heavy": 0.9,
        "orderby_heavy": 1.0,
        "distinct_heavy": 0.9,
        "subquery_heavy": 1.0,
        "setop_heavy": 0.8,
        "loader_heavy": 0.75,
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

        weights["join_heavy"] += 2.6 * join_gap + 2.2 * multi_join_gap + 1.6 * right_proj_gap
        weights["relationship_heavy"] += 2.4 * join_gap + 2.2 * right_proj_gap + 1.8 * left_join_gap
        weights["entity_heavy"] += 2.8 * entity_gap + 2.2 * entity_scalar_gap + 1.2 * join_gap
        weights["self_join_heavy"] += 3.2 * self_join_gap + 1.2 * orderby_gap
        weights["groupby_heavy"] += 2.8 * groupby_gap + 1.8 * having_gap + 1.8 * orderby_agg_gap
        weights["duplicate_column_heavy"] += 2.4 * duplicate_gap + 1.6 * distinct_gap
        weights["null_heavy"] += 2.6 * null_gap + 2.2 * left_join_gap + 1.8 * left_combo_gap
        weights["orderby_heavy"] += 3.0 * orderby_gap + 1.4 * right_proj_gap + 1.2 * orderby_agg_gap
        weights["distinct_heavy"] += 2.8 * distinct_gap + 1.6 * duplicate_gap
        weights["subquery_heavy"] += 3.0 * subquery_gap + 1.8 * exists_gap + 1.8 * in_subquery_gap
        weights["setop_heavy"] += 3.0 * set_query_gap + 1.4 * subquery_gap
        weights["loader_heavy"] += 2.4 * entity_gap + 2.0 * distinct_order_limit_gap + 1.2 * join_gap
        weights["orm_combo_heavy"] += 2.6 * entity_gap + 2.4 * set_query_gap + 2.0 * self_join_gap
        weights["combo_heavy"] += (
            2.6 * multi_join_gap
            + 2.4 * distinct_order_limit_gap
            + 1.8 * subquery_gap
            + 1.8 * left_combo_gap
            + 1.6 * orderby_agg_gap
        )

        if total < 50:
            weights["balanced"] *= 0.7
            weights["join_heavy"] += 0.6
            weights["relationship_heavy"] += 0.4
            weights["entity_heavy"] += 0.5
            weights["setop_heavy"] += 0.3
            weights["groupby_heavy"] += 0.4
            weights["orderby_heavy"] += 0.4
            weights["subquery_heavy"] += 0.5
            weights["loader_heavy"] += 0.3
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
    parser = argparse.ArgumentParser(description="RetORM 差分测试框架")
    parser.add_argument("--schemas",  type=int,  default=config.NUM_SCHEMAS)
    parser.add_argument("--queries",  type=int,  default=config.QUERIES_PER_SCHEMA)
    parser.add_argument("--tables",   type=int,  default=2)
    parser.add_argument("--cols",     type=int,  default=3)
    parser.add_argument("--rows",     type=int,  default=config.RANDOM_ROWS)
    parser.add_argument("--seed",     type=int,  default=None)
    parser.add_argument("--no-z3",   action="store_true")
    parser.add_argument("--verbose",  action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        num_schemas        = args.schemas,
        queries_per_schema = args.queries,
        num_tables         = args.tables,
        cols_per_table     = args.cols,
        rows_per_table     = args.rows,
        use_z3             = not args.no_z3,
        seed               = args.seed,
        verbose            = args.verbose,
    )
