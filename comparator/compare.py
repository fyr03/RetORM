"""
comparator/compare.py

结果比较器。

职责：
  1. 规范化：把三条路径返回的结果统一成可比较的格式
  2. 比较：做语义等价判断，输出是否一致以及差异详情

需要处理的差异：
  列名差异   : "o.amount" vs "amount"（python_ref 带前缀，sql/orm 不带）
  类型差异   : Decimal vs float、int vs float、bool vs int
  NULL 差异  : None / NaN 统一为 None
  顺序差异   : 无 ORDER BY 时结果顺序不确定，比较前排序
  浮点误差   : 聚合计算结果用容差比较
"""

import math
import re
import sys
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, __file__.rsplit("/comparator", 1)[0])


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

Row      = Dict[str, Any]
Rows     = List[Row]
NormRow  = Dict[str, Any]   # 规范化后的行
NormRows = List[NormRow]


# ---------------------------------------------------------------------------
# 公共配置
# ---------------------------------------------------------------------------

FLOAT_TOLERANCE      = 1e-6   # 普通浮点比较容差
AGG_FLOAT_TOLERANCE  = 1e-4   # 聚合函数结果容差（MySQL AVG 只保留4位小数）

# MySQL AVG 返回的 Decimal 只有4位小数精度（如 7.3333），
# 而 Python 用完整 float 精度计算（7.333333...），
# 两者差值约为 3e-5，需要用更宽松的容差。
# 识别方式：Decimal 类型说明值来自 MySQL 聚合，用宽松容差。


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

class CompareResult:
    """比较结果，携带是否一致和差异信息。"""

    def __init__(self, match: bool, reason: str = "", details: dict = None):
        self.match   = match
        self.reason  = reason
        self.details = details or {}

    def __bool__(self):
        return self.match

    def __repr__(self):
        if self.match:
            return "CompareResult(✓ match)"
        return f"CompareResult(✗ mismatch: {self.reason})"


def compare_all(
    ref_rows:  Rows,   # python_ref 路径的结果
    sql_rows:  Rows,   # raw sql 路径的结果
    orm_rows:  Rows,   # sqlalchemy_orm 路径的结果
    ordered:   bool = False,  # 是否有 ORDER BY（目前都是 False）
) -> Tuple[CompareResult, CompareResult]:
    """
    三路比较入口。

    返回两个 CompareResult：
      (ref_vs_sql, ref_vs_orm)

    以 python_ref 为基准，分别和 sql、orm 比较。
    如果 ref_vs_sql 不一致，说明 sql 翻译器有 bug（先排查这里）。
    如果 ref_vs_sql 一致但 ref_vs_orm 不一致，说明 ORM 有 bug。
    """
    norm_ref = normalize(ref_rows)
    norm_sql = normalize(sql_rows)
    norm_orm = normalize(orm_rows)

    ref_vs_sql = _compare_two(norm_ref, norm_sql, ordered, "ref", "sql")
    ref_vs_orm = _compare_two(norm_ref, norm_orm, ordered, "ref", "orm")

    return ref_vs_sql, ref_vs_orm


def compare_two_paths(
    rows_a: Rows,
    rows_b: Rows,
    name_a: str = "A",
    name_b: str = "B",
    ordered: bool = False,
) -> CompareResult:
    """
    只比较两条路径，方便单独调试。
    """
    norm_a = normalize(rows_a)
    norm_b = normalize(rows_b)
    return _compare_two(norm_a, norm_b, ordered, name_a, name_b)


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------

def normalize(rows: Rows) -> NormRows:
    """
    把一组结果行规范化：
      1. 每行的列名去掉表前缀（"o.amount" → "amount"）
      2. 每行的值做类型统一（Decimal→float、bool→int、NaN→None）
      3. 列名排序（保证同一行不同列顺序不影响比较）
    """
    return [_normalize_row(row) for row in rows]


def _normalize_row(row: Row) -> NormRow:
    """
    规范化一行数据的列名和值。

    冲突处理：如果同一行里多个字段规范化后得到同一个短名，
    就按“第几次出现”补上稳定编号，避免覆盖，也避免三条路径因为
    原始列名格式不同而无法对齐。

    例：
      ref 路径: {'o.id': 1, 'u.id': 2}
      sql 路径: {'id': 1, 'u.id': 2}
      orm 路径: {'o_id': 1, 'u_id': 2}

    三者都会规范化为:
      {'id#1': 1, 'id#2': 2}
    """
    # 第一遍：按原始列顺序收集规范化短名，并统计出现次数
    norm_keys_in_order: List[str] = []
    norm_key_count: Dict[str, int] = {}
    for key in row.keys():
        nk = _normalize_key(key)
        norm_keys_in_order.append(nk)
        norm_key_count[nk] = norm_key_count.get(nk, 0) + 1

    # 第二遍：重复短名按出现次序编号，唯一短名保持不变
    result = {}
    seen_count: Dict[str, int] = {}
    for (key, val), norm_key in zip(row.items(), norm_keys_in_order):
        norm_val = _normalize_value(val)
        if norm_key_count[norm_key] > 1:
            seen_count[norm_key] = seen_count.get(norm_key, 0) + 1
            canonical_key = f"{norm_key}#{seen_count[norm_key]}"
        else:
            canonical_key = norm_key
        result[canonical_key] = norm_val
    return result


# 别名前缀模式：1个字母 + 可选数字，如 o / p / p2 / u3
_ALIAS_PREFIX_RE = re.compile(r'^([a-zA-Z]\d*)_(.+)$')


def _normalize_key(key: str) -> str:
    """
    把列名统一成不带表前缀的短名。

    三条路径的列名格式：
      ref 路径  : "o.price"   带 "." 的表别名前缀
                  "p2.num"    双字符别名（字母+数字）
      sql 路径  : "price"     MySQL 原始列名
                  "p2.num"    两表同名列时 MySQL 保留前缀
      orm 路径  : "o_price"   label 把 "." 替换成 "_"（单字母别名）
                  "p2_num"    双字符别名（字母+数字）

    统一规则：
      1. 含 "."  → 取最后一段
      2. 含 "_"  → 首段匹配 "字母+可选数字"（如 o/p/p2/u3）→ 去掉首段
                   否则保持原样（products_id / count_all 等真实列名不动）
    """
    # 规则 1：含 "."
    if "." in key:
        return key.split(".", 1)[1]

    # 规则 2：orm label 格式，首段是"字母+可选数字"的别名
    if "_" in key:
        m = _ALIAS_PREFIX_RE.match(key)
        if m:
            return m.group(2)

    return key


class AggFloat(float):
    """
    标记"来自数据库聚合函数（AVG/SUM等）的浮点数"。
    用于在 _values_equal 里使用更宽松的容差。
    MySQL AVG 只保留4位小数，Python 保留完整精度，差值约 3e-5。
    """
    pass


def _normalize_value(val: Any) -> Any:
    """
    统一值类型：
      Decimal → AggFloat（标记为聚合结果，使用宽松容差）
      bool    → int
      NaN     → None
      其余保持不变
    """
    if val is None:
        return None
    if isinstance(val, Decimal):
        # 用 AggFloat 而不是普通 float，以便比较时用宽松容差
        return AggFloat(val)
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


# ---------------------------------------------------------------------------
# 核心比较逻辑
# ---------------------------------------------------------------------------

def _compare_two(
    norm_a:  NormRows,
    norm_b:  NormRows,
    ordered: bool,
    name_a:  str,
    name_b:  str,
) -> CompareResult:
    """
    比较两组规范化后的结果。
    """
    # 1. 行数必须相同
    if len(norm_a) != len(norm_b):
        return CompareResult(
            match=False,
            reason=f"行数不同：{name_a}={len(norm_a)} 行，{name_b}={len(norm_b)} 行",
            details={
                name_a: norm_a,
                name_b: norm_b,
            }
        )

    if len(norm_a) == 0:
        # 两者都是空结果，视为一致
        return CompareResult(match=True, reason="两者都返回空结果")

    # 2. 列名集合必须相同
    keys_a = set(norm_a[0].keys())
    keys_b = set(norm_b[0].keys())
    if keys_a != keys_b:
        return CompareResult(
            match=False,
            reason=f"列名不同：{name_a}={sorted(keys_a)}，{name_b}={sorted(keys_b)}",
            details={"keys_a": sorted(keys_a), "keys_b": sorted(keys_b)}
        )

    # 3. 排序（无 ORDER BY 时，顺序不确定，排序后比较）
    if not ordered:
        sort_keys = sorted(keys_a)
        norm_a = _sort_rows(norm_a, sort_keys)
        norm_b = _sort_rows(norm_b, sort_keys)

    # 4. 逐行逐列比较
    for i, (row_a, row_b) in enumerate(zip(norm_a, norm_b)):
        for col in sorted(row_a.keys()):
            val_a = row_a[col]
            val_b = row_b.get(col)
            if not _values_equal(val_a, val_b):
                return CompareResult(
                    match=False,
                    reason=f"第 {i+1} 行，列 '{col}' 不一致："
                           f"{name_a}={val_a!r}，{name_b}={val_b!r}",
                    details={
                        "row_index": i,
                        "column":    col,
                        name_a:      row_a,
                        name_b:      row_b,
                    }
                )

    return CompareResult(match=True)


def _sort_rows(rows: NormRows, sort_keys: List[str]) -> NormRows:
    """
    按指定列排序，用于无 ORDER BY 时的结果对比。
    None 值排在最前（统一处理，避免排序时 TypeError）。
    """
    def sort_key(row):
        return tuple(
            (0, "")    if row.get(k) is None
            else (1, row[k]) if isinstance(row.get(k), str)
            else (1, row[k])
            for k in sort_keys
        )
    return sorted(rows, key=sort_key)


def _values_equal(a: Any, b: Any) -> bool:
    """
    比较两个值是否语义相等。
    - None == None → True
    - AggFloat（来自 Decimal）用宽松容差（MySQL AVG 精度损失约 3e-5）
    - 普通 float/int 用严格容差
    - 其余用 ==
    """
    # 两者都是 None
    if a is None and b is None:
        return True

    # 一个 None 一个不是
    if a is None or b is None:
        return False

    # bool 精确比较（bool 是 int 的子类，要先判断）
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b

    # 数值类型
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        # 任一方是 AggFloat（来自 Decimal 聚合结果），用宽松容差
        if isinstance(a, AggFloat) or isinstance(b, AggFloat):
            return math.isclose(fa, fb,
                                rel_tol=AGG_FLOAT_TOLERANCE,
                                abs_tol=AGG_FLOAT_TOLERANCE)
        return math.isclose(fa, fb,
                            rel_tol=FLOAT_TOLERANCE,
                            abs_tol=FLOAT_TOLERANCE)

    return a == b


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def print_report(
    ref_vs_sql: CompareResult,
    ref_vs_orm: CompareResult,
    ir_desc:    str = "",
) -> None:
    """
    把比较结果打印成人类可读的报告。
    """
    print("\n" + "=" * 60)
    if ir_desc:
        print(f"IR: {ir_desc}")

    print(f"[ref vs sql] {'✓ 一致' if ref_vs_sql.match else '✗ 不一致'}")
    if not ref_vs_sql.match:
        print(f"  原因: {ref_vs_sql.reason}")
        _print_details(ref_vs_sql.details)

    print(f"[ref vs orm] {'✓ 一致' if ref_vs_orm.match else '✗ 不一致'}")
    if not ref_vs_orm.match:
        print(f"  原因: {ref_vs_orm.reason}")
        _print_details(ref_vs_orm.details)

    if ref_vs_sql.match and ref_vs_orm.match:
        print("→ 三路结果一致，未发现 bug")
    elif not ref_vs_sql.match:
        print("→ ref vs sql 不一致，优先排查 sql 翻译器")
    else:
        print("→ ref vs orm 不一致，ORM 可能存在 bug ⚠️")
    print("=" * 60)


def _print_details(details: dict) -> None:
    for k, v in details.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"    {k}: (共 {len(v)} 行，前 5 行) {v[:5]}")
        else:
            print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# 直接运行此文件时：端到端三路比较测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from db.connector import (
        create_tables, insert_rows, drop_tables,
        init_database, dispose_engine,
    )
    from ir.nodes import (Aggregate, AggFunc, CmpOp, pretty_print,
                          Scan, Filter, Join, GroupBy, Having, Project, Compare)
    from translators.python_ref import execute as ref_execute
    from translators.sql         import execute as sql_execute
    from translators.sqlalchemy_orm import execute as orm_execute, reset_metadata

    print("=== 准备数据库 ===")
    init_database()
    drop_tables(["orders", "users"])
    create_tables([
        """CREATE TABLE IF NOT EXISTS `users` (
            `id` INT PRIMARY KEY, `age` INT NOT NULL
        );""",
        """CREATE TABLE IF NOT EXISTS `orders` (
            `id` INT PRIMARY KEY,
            `user_id` INT NOT NULL,
            `amount` FLOAT NOT NULL
        );""",
    ])
    insert_rows("users",  ["id", "age"], [(1, 25), (2, 17), (3, 30)])
    insert_rows("orders", ["id", "user_id", "amount"],
                [(1, 1, 50.0), (2, 1, 80.0), (3, 2, 200.0), (4, 3, 30.0)])
    reset_metadata()

    # ------------------------------------------------------------------
    # 测试一：单表 Filter + Project
    # ------------------------------------------------------------------
    print("\n=== 测试一：单表 Filter + Project ===")
    ir1 = Project(
        fields=["user_id"],
        child=Filter(
            condition=Compare("amount", CmpOp.GT, 60),
            child=Scan("orders")
        )
    )
    ref1 = ref_execute(ir1)
    sql1 = sql_execute(ir1)
    orm1 = orm_execute(ir1)
    print(f"ref: {ref1}")
    print(f"sql: {sql1}")
    print(f"orm: {orm1}")
    r1, r2 = compare_all(ref1, sql1, orm1)
    print_report(r1, r2, "Filter + Project")
    assert r1.match and r2.match, "测试一失败"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试二：GroupBy + Having + Project
    # ------------------------------------------------------------------
    print("\n=== 测试二：GroupBy + Having + Project ===")
    ir2 = Project(
        fields=["user_id", "total"],
        child=Having(
            condition=Compare("total", CmpOp.GT, 100),
            child=GroupBy(
                fields=["user_id"],
                aggregates=[Aggregate(AggFunc.SUM, "amount", "total")],
                child=Scan("orders")
            )
        )
    )
    ref2 = ref_execute(ir2)
    sql2 = sql_execute(ir2)
    orm2 = orm_execute(ir2)
    print(f"ref: {ref2}")
    print(f"sql: {sql2}")
    print(f"orm: {orm2}")
    r1, r2 = compare_all(ref2, sql2, orm2)
    print_report(r1, r2, "GroupBy + Having + Project")
    assert r1.match and r2.match, "测试二失败"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试三：Join + Filter + Project
    # ------------------------------------------------------------------
    print("\n=== 测试三：Join + Filter + Project ===")
    ir3 = Project(
        fields=["o.user_id", "o.amount"],
        child=Filter(
            condition=Compare("u.age", CmpOp.GT, 18),
            child=Join(
                left=Scan("orders", "o"),
                right=Scan("users", "u"),
                on=Compare("o.user_id", CmpOp.EQ, "u.id")
            )
        )
    )
    ref3 = ref_execute(ir3)
    sql3 = sql_execute(ir3)
    orm3 = orm_execute(ir3)
    print(f"ref: {ref3}")
    print(f"sql: {sql3}")
    print(f"orm: {orm3}")
    r1, r2 = compare_all(ref3, sql3, orm3)
    print_report(r1, r2, "Join + Filter + Project")
    assert r1.match and r2.match, "测试三失败"
    print("✓ 通过")

    # 清理
    print("\n=== 清理 ===")
    drop_tables(["orders", "users"])
    dispose_engine()
    print("\n全部测试通过 ✓")
