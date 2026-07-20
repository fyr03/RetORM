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
from translators.sqlalchemy_orm import execute as orm_execute, reset_metadata
from comparator.compare         import compare_all, print_report
from db.connector import (
    init_database, create_tables, drop_tables,
    execute_sql, dispose_engine,
)
from ir.nodes import pretty_print, Scan, Filter, Join, GroupBy, Having, Project


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
        bug_total   = s.sql_bugs + s.orm_bugs
        bug_rate    = f"{bug_total / total * 100:.1f}%" if total else "N/A"

        # 速度：queries per minute
        qpm = f"{total / elapsed * 60:.1f}" if elapsed > 0 else "N/A"

        lines = [
            f"[运行统计]  耗时={elapsed}s  速度={qpm} q/min , 查询总数  : {total}",
            f"  通过      : {s.passed}  ({pass_rate}),  空结果    : {s.empty_results}  ({empty_rate}),  执行错误  : {s.errors}  ({error_rate})",
            f"  SQL bug   : {s.sql_bugs},  ORM bug   : {s.orm_bugs},  bug 合计  : {bug_total}  ({bug_rate})",
            f"  结构覆盖  : 单表={s.single_table_queries}, Join={s.join_queries}, Filter={s.filter_queries}, GroupBy={s.groupby_queries}, Having={s.having_queries}, 重复投影={s.duplicate_proj_queries}",
        ]
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
    ref_vs_orm:   object          # CompareResult
    ref_rows:     list
    sql_rows:     list
    orm_rows:     list
    sql_text:     str = ""        # 生成的 SQL 字符串
    error:        Optional[str] = None


@dataclass
class RunStats:
    total_queries: int = 0
    passed:        int = 0
    sql_bugs:      int = 0
    orm_bugs:      int = 0
    errors:        int = 0
    empty_results: int = 0
    single_table_queries:   int = 0
    join_queries:           int = 0
    filter_queries:         int = 0
    groupby_queries:        int = 0
    having_queries:         int = 0
    duplicate_proj_queries: int = 0
    bug_reports:   List[BugReport] = field(default_factory=list)


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
    if report.error:
        bug_type = "执行异常"
        reason   = report.error
    elif (report.ref_vs_sql and not report.ref_vs_sql.match and
          report.ref_vs_orm and not report.ref_vs_orm.match):
        bug_type = "ref 路径可能有 bug（ref vs sql 和 ref vs orm 均不一致）"
        reason   = (f"ref_vs_sql: {report.ref_vs_sql.reason} | "
                    f"ref_vs_orm: {report.ref_vs_orm.reason}")
    elif report.ref_vs_sql and not report.ref_vs_sql.match:
        bug_type = "SQL 翻译器 bug"
        reason   = report.ref_vs_sql.reason
    elif report.ref_vs_orm and not report.ref_vs_orm.match:
        bug_type = "ORM bug"
        reason   = report.ref_vs_orm.reason
    else:
        bug_type = "未知（三路均一致，不应出现）"
        reason   = ""

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
        "── ORM API（路径三：SQLAlchemy Core）──",
        "  通过 translators/sqlalchemy_orm.py 的 execute(ir) 执行，",
        "  SQLAlchemy 根据 IR 动态构建 Select 对象，经过以下步骤：",
        "    1. _collect_ctx(ir)  递归收集 FROM / JOIN / WHERE / GROUP BY / HAVING",
        "    2. _assemble(ctx)    组装成 SQLAlchemy Select 语句",
        "    3. engine.connect().execute(stmt)  提交给 MySQL 执行",
        "  等价于以下 SQLAlchemy Core 调用链（伪代码）：",
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
        f"  orm（ORM API） {len(report.orm_rows)} 行: {report.orm_rows}",
        "",
    ]

    if report.ref_vs_sql and not report.ref_vs_sql.match:
        lines.append(f"  ref vs sql 不一致: {report.ref_vs_sql.reason}")
    if report.ref_vs_orm and not report.ref_vs_orm.match:
        lines.append(f"  ref vs orm 不一致: {report.ref_vs_orm.reason}")

    lines.append(sep)

    bug_logger.info("\n".join(lines))
    # 关闭这个 logger 的 handler，避免重复写入
    bh.close()
    bug_logger.removeHandler(bh)
    return fpath_detail


def _gen_orm_pseudocode(report: BugReport) -> str:
    """根据 IR 生成 SQLAlchemy Core 的伪代码描述。"""
    from ir.nodes import Scan, Filter, Join, GroupBy, Having, Project

    # 直接把 IR 结构转成伪代码注释，简单可靠
    lines = ["    # IR → SQLAlchemy Core 伪代码："]
    for line in report.ir_str.split("\n"):
        lines.append("    # " + line)
    return "\n".join(lines)


def _write_bug_file(report: BugReport, bug_dir: str, bug_idx: int) -> str:
    """
    把一个 bug 写成可直接运行的 Python 复现脚本。
    返回文件路径。
    """
    os.makedirs(bug_dir, exist_ok=True)
    fpath = os.path.join(bug_dir, f"bug_{bug_idx:03d}.py")

    # ── 确定 bug 类型 ──
    if report.error:
        bug_type = "执行异常"
        reason   = report.error
    elif (report.ref_vs_sql and not report.ref_vs_sql.match and
          report.ref_vs_orm and not report.ref_vs_orm.match):
        bug_type = "ref 路径可能有 bug（ref vs sql 和 ref vs orm 均不一致）"
        reason   = (f"ref_vs_sql: {report.ref_vs_sql.reason} | "
                    f"ref_vs_orm: {report.ref_vs_orm.reason}")
    elif report.ref_vs_sql and not report.ref_vs_sql.match:
        bug_type = "SQL 翻译器 bug"
        reason   = report.ref_vs_sql.reason
    elif report.ref_vs_orm and not report.ref_vs_orm.match:
        bug_type = "ORM bug ⚠️"
        reason   = report.ref_vs_orm.reason
    else:
        bug_type = "未知（三路均一致，不应出现）"
        reason   = ""

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
    from translators.sqlalchemy_orm import execute as orm_execute, reset_metadata
    from comparator.compare         import compare_all, print_report
    from ir.nodes import *

    # ── 1. 建表 ────────────────────────────────────────────────────────────
    init_database()
    drop_tables({drop_order!r})
    create_tables({create_sqls!r})
    reset_metadata()

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
    orm_rows = orm_execute(ir)

    print(f"\\nref 结果 ({{len(ref_rows)}} 行): {{ref_rows}}")
    print(f"sql 结果 ({{len(sql_rows)}} 行): {{sql_rows}}")
    print(f"orm 结果 ({{len(orm_rows)}} 行): {{orm_rows}}")

    # ── 6. 比较 ────────────────────────────────────────────────────────────
    ref_vs_sql, ref_vs_orm = compare_all(ref_rows, sql_rows, orm_rows)
    print_report(ref_vs_sql, ref_vs_orm)

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
    from ir.nodes import Scan, Filter, Join, GroupBy, Having, Project
    from ir.nodes import Compare, And, Or, Not, Aggregate

    if isinstance(node, Scan):
        return f"Scan(table={node.table!r}, alias={node.alias!r})"
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
            f"GroupBy(fields={node.fields!r}, "
            f"aggregates=[{aggs}], "
            f"child={_gen_ir_code(node.child)})"
        )
    if isinstance(node, Having):
        return (
            f"Having(condition={_gen_ir_code(node.condition)}, "
            f"child={_gen_ir_code(node.child)})"
        )
    if isinstance(node, Project):
        return f"Project(fields={node.fields!r}, child={_gen_ir_code(node.child)})"
    if isinstance(node, Compare):
        return (
            f"Compare(field={node.field!r}, op=CmpOp.{node.op.name}, "
            f"value={node.value!r})"
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
            f"field={node.field!r}, alias={node.alias!r})"
        )
    raise TypeError(f"不支持序列化的 IR 节点: {type(node)}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(
    num_schemas:        int   = config.NUM_SCHEMAS,
    queries_per_schema: int   = config.QUERIES_PER_SCHEMA,
    num_tables:         int   = 2,
    cols_per_table:     int   = 3,
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
        f"  tables={num_tables}, cols={cols_per_table}\n"
        f"  use_z3={use_z3}, seed={seed}\n"
        f"  detail_log=detail_logs/{run_ts}.log\n"
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
    run_logger.info(f"  RANDOM_ROWS={config.RANDOM_ROWS}  Z3_TIMEOUT={config.Z3_TIMEOUT_SEC}s")
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
                reset_metadata()
            except Exception as e:
                msg = f"[runner] 建表失败，跳过此 schema: {e}"
                print(msg); dlog(msg)
                continue

            # ── 每条 IR ──────────────────────────────────────────────────
            for query_id in range(queries_per_schema):
                query_seed = schema_seed + query_id + 1
                stats.total_queries += 1
                table_data = {}
                stress_mode = _choose_stress_mode(query_seed)

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

                _record_query_shape(stats, ir)

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
                        rows_per_table=config.RANDOM_ROWS,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
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
                    ref_rows = ref_execute(ir)
                    sql_rows = sql_execute(ir)
                    orm_rows = orm_execute(ir)
                    sql_text = sql_translate(ir)

                    if verbose:
                        exec_info = (
                            f"  ref: {ref_rows}\n"
                            f"  sql: {sql_rows}\n"
                            f"  orm: {orm_rows}\n"
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
                    bug_idx += 1
                    report = BugReport(
                        schema_id=schema_id, query_id=query_id,
                        schema=schema, ir=ir, ir_str=ir_str,
                        schema_seed=schema_seed, query_seed=query_seed,
                        table_data=table_data,
                        rows_per_table=config.RANDOM_ROWS,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        ref_vs_sql=None, ref_vs_orm=None,
                        ref_rows=[], sql_rows=[], orm_rows=[],
                        error=str(e),
                    )
                    stats.bug_reports.append(report)
                    fpath = _write_bug_file(report, bug_dir, bug_idx)
                    fpath_detail = _write_bug_detail(report)
                    msg2 = f"  → 复现脚本: {fpath}  bug详情: {fpath_detail}"
                    print(msg2); dlog(msg2)
                    continue

                # 6. 比较
                try:
                    ref_vs_sql, ref_vs_orm = compare_all(ref_rows, sql_rows, orm_rows)
                except Exception as e:
                    msg = f"[比较失败] {e}"
                    print(msg); dlog(msg)
                    stats.errors += 1
                    continue

                # 7. 统计 & 日志
                all_empty = (len(ref_rows) == 0 and
                             len(sql_rows) == 0 and
                             len(orm_rows) == 0)
                if all_empty:
                    print("(空结果)", end="")
                    stats.empty_results += 1

                if ref_vs_sql.match and ref_vs_orm.match:
                    print("✓")
                    # 通过时一行总结，verbose 时附上三路结果
                    if verbose:
                        dlog(f"  ref({len(ref_rows)}行) sql({len(sql_rows)}行) "
                             f"orm({len(orm_rows)}行)  → ✓ 三路一致"
                             + ("  [空结果]" if all_empty else ""))
                    else:
                        dlog(f"  → ✓ 三路一致"
                             + ("  [空结果]" if all_empty else ""))
                    stats.passed += 1
                else:
                    if not ref_vs_sql.match:
                        tag = "✗ SQL bug"
                        stats.sql_bugs += 1
                    else:
                        tag = "✗ ORM bug ⚠️"
                        stats.orm_bugs += 1

                    print(tag)
                    dlog(f"  ref({len(ref_rows)}行) sql({len(sql_rows)}行) "
                         f"orm({len(orm_rows)}行)  → {tag}")

                    # 写详细比较到详细日志
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        print_report(ref_vs_sql, ref_vs_orm, ir_str)
                    report_str = buf.getvalue()
                    if verbose:
                        print(report_str)
                    dlog(report_str)

                    bug_idx += 1
                    report = BugReport(
                        schema_id=schema_id, query_id=query_id,
                        schema=schema, ir=ir, ir_str=ir_str,
                        schema_seed=schema_seed, query_seed=query_seed,
                        table_data=table_data,
                        rows_per_table=config.RANDOM_ROWS,
                        use_z3=use_z3,
                        z3_timeout=config.Z3_TIMEOUT_SEC,
                        ref_vs_sql=ref_vs_sql, ref_vs_orm=ref_vs_orm,
                        ref_rows=ref_rows, sql_rows=sql_rows, orm_rows=orm_rows,
                        sql_text=sql_text,
                    )
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
    run_logger.info(f"  ORM bug   : {s.orm_bugs}  ← 重点关注")
    run_logger.info(f"  bug 合计  : {s.sql_bugs + s.orm_bugs}")
    run_logger.info(
        "  结构覆盖  : "
        f"单表={s.single_table_queries}  Join={s.join_queries}  "
        f"Filter={s.filter_queries}  GroupBy={s.groupby_queries}  "
        f"Having={s.having_queries}  重复投影={s.duplicate_proj_queries}"
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


def _choose_stress_mode(query_seed: int) -> str:
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


def _record_query_shape(stats: RunStats, ir) -> None:
    features = _collect_query_features(ir)

    if features["has_join"]:
        stats.join_queries += 1
    else:
        stats.single_table_queries += 1

    if features["has_filter"]:
        stats.filter_queries += 1
    if features["has_groupby"]:
        stats.groupby_queries += 1
    if features["has_having"]:
        stats.having_queries += 1
    if features["has_duplicate_projection"]:
        stats.duplicate_proj_queries += 1


def _collect_query_features(node) -> dict:
    features = {
        "has_join": False,
        "has_filter": False,
        "has_groupby": False,
        "has_having": False,
        "has_duplicate_projection": False,
    }

    def visit(cur):
        if isinstance(cur, Join):
            features["has_join"] = True
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, Filter):
            features["has_filter"] = True
            visit(cur.child)
            return
        if isinstance(cur, GroupBy):
            features["has_groupby"] = True
            visit(cur.child)
            return
        if isinstance(cur, Having):
            features["has_having"] = True
            visit(cur.child)
            return
        if isinstance(cur, Project):
            short_names = [_short_field_name(field) for field in cur.fields]
            features["has_duplicate_projection"] = len(short_names) != len(set(short_names))
            visit(cur.child)
            return
        if isinstance(cur, Scan):
            return

    visit(node)
    return features


def _short_field_name(field_name: str) -> str:
    return field_name.split(".", 1)[1] if "." in field_name else field_name


def _print_final_report(stats: RunStats, bug_dir: str = "bugs") -> None:
    print("\n" + "=" * 60)
    print("测试完成")
    print(f"  总查询数    : {stats.total_queries}")
    print(f"  通过        : {stats.passed}")
    print(f"  空结果      : {stats.empty_results}")
    print(f"  执行错误    : {stats.errors}")
    print(f"  SQL 翻译 bug: {stats.sql_bugs}")
    print(f"  ORM bug     : {stats.orm_bugs}  ← 重点关注")
    print("  结构覆盖    :")
    print(f"    单表      : {stats.single_table_queries}")
    print(f"    Join      : {stats.join_queries}")
    print(f"    Filter    : {stats.filter_queries}")
    print(f"    GroupBy   : {stats.groupby_queries}")
    print(f"    Having    : {stats.having_queries}")
    print(f"    重复投影  : {stats.duplicate_proj_queries}")
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
            if (r.ref_vs_sql and not r.ref_vs_sql.match and
                    r.ref_vs_orm and not r.ref_vs_orm.match):
                print(f"  类型: ref 路径可能有 bug")
                print(f"  原因(sql): {r.ref_vs_sql.reason}")
                print(f"  原因(orm): {r.ref_vs_orm.reason}")
            elif r.ref_vs_sql and not r.ref_vs_sql.match:
                print(f"  类型: SQL 翻译器 bug")
                print(f"  原因: {r.ref_vs_sql.reason}")
            elif r.ref_vs_orm and not r.ref_vs_orm.match:
                print(f"  类型: ORM bug ⚠️")
                print(f"  原因: {r.ref_vs_orm.reason}")
            print(f"  ref ({len(r.ref_rows)}行): {r.ref_rows[:3]}"
                  f"{'...' if len(r.ref_rows) > 3 else ''}")
            print(f"  sql ({len(r.sql_rows)}行): {r.sql_rows[:3]}"
                  f"{'...' if len(r.sql_rows) > 3 else ''}")
            print(f"  orm ({len(r.orm_rows)}行): {r.orm_rows[:3]}"
                  f"{'...' if len(r.orm_rows) > 3 else ''}")
        print(f"  IR:\n{r.ir_str}")
        print(f"  复现: {os.path.join(bug_dir, f'bug_{i+1:03d}.py')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="RetORM 差分测试框架")
    parser.add_argument("--schemas",  type=int,  default=config.NUM_SCHEMAS)
    parser.add_argument("--queries",  type=int,  default=config.QUERIES_PER_SCHEMA)
    parser.add_argument("--tables",   type=int,  default=2)
    parser.add_argument("--cols",     type=int,  default=3)
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
        use_z3             = not args.no_z3,
        seed               = args.seed,
        verbose            = args.verbose,
    )
