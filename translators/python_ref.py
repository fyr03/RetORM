"""
translators/python_ref.py

路径一：程序逻辑参照实现。

工作方式：
  1. 遍历 IR 树，找出所有涉及的表
  2. 用 SELECT * 把原始数据全部取出来
  3. 在 Python 里手动完成 join / filter / group / aggregate / having / project

原则：
  - 逻辑越简单越好，简单到一眼能看出正确性
  - 不做任何优化
  - 这条路径的结果是 differential oracle 的基准
"""

import sys
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, __file__.rsplit("/translators", 1)[0])

from ir.nodes import (
    Scan, Filter, Join, GroupBy, Having, Project,
    Compare, And, Or, Not, Aggregate,
    AggFunc, CmpOp, JoinType, QueryNode, Condition,
)
from db.connector import execute_sql


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

Row  = Dict[str, Any]   # 一行数据，key 是 "table.column" 或 "column"
Rows = List[Row]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def execute(ir: QueryNode) -> Rows:
    """
    接收一棵 IR 树，返回执行结果。
    结果是 List[Dict]，每个 dict 是一行，key 是列名。

    用法：
        from translators.python_ref import execute
        rows = execute(my_ir)
    """
    return _eval(ir)


# ---------------------------------------------------------------------------
# 递归求值
# ---------------------------------------------------------------------------

def _eval(node: QueryNode) -> Rows:
    """递归处理 IR 树，返回当前节点对应的行集合。"""

    if isinstance(node, Scan):
        return _eval_scan(node)

    elif isinstance(node, Filter):
        return _eval_filter(node)

    elif isinstance(node, Join):
        return _eval_join(node)

    elif isinstance(node, GroupBy):
        return _eval_groupby(node)

    elif isinstance(node, Having):
        return _eval_having(node)

    elif isinstance(node, Project):
        return _eval_project(node)

    else:
        raise NotImplementedError(f"未知节点类型: {type(node)}")


# ---------------------------------------------------------------------------
# Scan：SELECT * FROM table，列名统一加表名前缀
# ---------------------------------------------------------------------------

def _eval_scan(node: Scan) -> Rows:
    """
    取出整张表的所有行。
    为了后续 join 时区分来自不同表的同名列，
    把所有列名统一改成 "alias.column" 格式。

    例：Scan("orders", "o")
        原始行: {"id": 1, "user_id": 1, "amount": 50.0}
        输出行: {"o.id": 1, "o.user_id": 1, "o.amount": 50.0}
    """
    raw_rows = execute_sql(f"SELECT * FROM `{node.table}`;")
    result = []
    for row in raw_rows:
        prefixed = {f"{node.alias}.{col}": val for col, val in row.items()}
        result.append(prefixed)
    return result


# ---------------------------------------------------------------------------
# Filter：在 Python 里逐行检查条件
# ---------------------------------------------------------------------------

def _eval_filter(node: Filter) -> Rows:
    rows = _eval(node.child)
    return [row for row in rows if _eval_condition(node.condition, row)]


# ---------------------------------------------------------------------------
# Join：嵌套循环 inner join，简单直接
# ---------------------------------------------------------------------------

def _eval_join(node: Join) -> Rows:
    """
    最朴素的嵌套循环 INNER JOIN。
    on 条件的右值如果是字符串且包含 "."，视为列引用。
    """
    left_rows  = _eval(node.left)
    right_rows = _eval(node.right)
    right_null_row = {key: None for key in right_rows[0].keys()} if right_rows else {}

    result = []
    for l_row in left_rows:
        matched = False
        for r_row in right_rows:
            # 合并两行
            merged = {**l_row, **r_row}
            # 检查 on 条件
            if _eval_condition(node.on, merged):
                matched = True
                result.append(merged)
        if node.join_type == JoinType.LEFT and not matched:
            result.append({**l_row, **right_null_row})
    return result


# ---------------------------------------------------------------------------
# GroupBy：按字段分组，计算聚合值，把聚合结果合并进每行
# ---------------------------------------------------------------------------

def _eval_groupby(node: GroupBy) -> Rows:
    """
    按 node.fields 分组，对每个组计算 node.aggregates 里的聚合函数。
    返回的每行包含：分组字段 + 所有聚合别名。

    例：GroupBy(fields=["o.user_id"], aggregates=[Aggregate(SUM, "o.amount", "total")])
        输入行: [{"o.user_id": 1, "o.amount": 50}, {"o.user_id": 1, "o.amount": 80}]
        输出行: [{"o.user_id": 1, "total": 130}]
    """
    rows = _eval(node.child)

    # 按分组字段把行分桶
    # key: 分组字段值的元组；value: 属于这个组的所有行
    #
    # SQL NULL 语义：GROUP BY 时所有 NULL 值归为同一组（NULL == NULL）。
    # Python 的 None 在 tuple key 里也满足这个语义（None == None），
    # 但 defaultdict 用 tuple 做 key 时 None 是合法且唯一的，
    # 和 SQL 行为一致，不需要特殊处理。
    # 注意：dict key 里不能有 float('nan')，需要提前规范化。
    groups: Dict[tuple, Rows] = defaultdict(list)
    for row in rows:
        raw_key = (_resolve_field(f, row) for f in node.fields)
        # 把 float NaN 规范化为 None，避免 NaN != NaN 导致分组错误
        group_key = tuple(
            None if (isinstance(v, float) and v != v) else v
            for v in raw_key
        )
        groups[group_key].append(row)

    result = []
    for group_key, group_rows in groups.items():
        # 构造输出行
        out_row: Row = {}

        # 填入分组字段的值
        for field_name, val in zip(node.fields, group_key):
            out_row[field_name] = val

        # 计算每个聚合函数
        for agg in node.aggregates:
            out_row[agg.alias] = _compute_aggregate(agg, group_rows)

        result.append(out_row)

    return result


def _compute_aggregate(agg: Aggregate, rows: Rows) -> Any:
    """对一组行计算单个聚合函数的值。"""
    if agg.func == AggFunc.COUNT:
        if agg.field == "*":
            return len(rows)
        else:
            # COUNT(field) 不计 NULL
            return sum(1 for r in rows if _resolve_field(agg.field, r) is not None)

    # 其余聚合函数需要取字段值列表（过滤 NULL）
    values = [_resolve_field(agg.field, r) for r in rows]
    values = [v for v in values if v is not None]

    if not values:
        return None

    if agg.func == AggFunc.SUM:
        return sum(values)
    elif agg.func == AggFunc.AVG:
        return sum(values) / len(values)
    elif agg.func == AggFunc.MAX:
        return max(values)
    elif agg.func == AggFunc.MIN:
        return min(values)
    else:
        raise NotImplementedError(f"未知聚合函数: {agg.func}")


# ---------------------------------------------------------------------------
# Having：对 GroupBy 结果按条件过滤
# ---------------------------------------------------------------------------

def _eval_having(node: Having) -> Rows:
    rows = _eval(node.child)  # child 必须是 GroupBy
    return [row for row in rows if _eval_condition(node.condition, row)]


# ---------------------------------------------------------------------------
# Project：只保留指定列
# ---------------------------------------------------------------------------

def _eval_project(node: Project) -> Rows:
    """
    只保留 node.fields 里指定的列。
    字段名支持 "table.column" 和 "column" 两种格式。
    """
    rows = _eval(node.child)
    result = []
    for row in rows:
        out_row: Row = {}
        for field_name in node.fields:
            out_row[field_name] = _resolve_field(field_name, row)
        result.append(out_row)
    return result


# ---------------------------------------------------------------------------
# 条件求值（三值逻辑：True / False / None）
#
# SQL 里 NULL 参与的任何比较结果都是 NULL（既不是 TRUE 也不是 FALSE）。
# WHERE / HAVING 只保留结果为 TRUE 的行，NULL 和 FALSE 都会被过滤。
#
# 三值逻辑规则：
#   NOT NULL  = NULL
#   NULL AND TRUE  = NULL,  NULL AND FALSE = FALSE,  NULL AND NULL = NULL
#   NULL OR  TRUE  = TRUE,  NULL OR  FALSE = NULL,   NULL OR  NULL = NULL
# ---------------------------------------------------------------------------

def _eval_condition(cond: Condition, row: Row) -> bool:
    """
    递归求值条件节点，返回 True/False。
    内部用三值逻辑（True/False/None），最终对外只返回 bool。
    """
    result = _eval_condition_3vl(cond, row)
    # None（SQL NULL）视为 False：不满足条件
    return result is True


def _eval_condition_3vl(cond: Condition, row: Row):
    """
    三值逻辑求值，返回 True / False / None（None 代表 SQL NULL）。
    """
    if isinstance(cond, Compare):
        left_val = _resolve_field(cond.field, row)

        # 右值：字符串且含 "." → 列引用；否则是字面量
        if isinstance(cond.value, str) and "." in cond.value:
            right_val = _resolve_field(cond.value, row)
        else:
            right_val = cond.value
        if right_val is None and not (isinstance(cond.value, str) and "." in str(cond.value)):
            if cond.op == CmpOp.EQ:
                return left_val is None
            if cond.op == CmpOp.NEQ:
                return left_val is not None

        # 任一操作数为 NULL → 结果为 NULL
        if left_val is None or right_val is None:
            return None
        return _compare(left_val, cond.op, right_val)

    elif isinstance(cond, And):
        left  = _eval_condition_3vl(cond.left,  row)
        right = _eval_condition_3vl(cond.right, row)
        # FALSE AND anything = FALSE
        if left is False or right is False:
            return False
        # NULL AND TRUE = NULL,  NULL AND NULL = NULL
        if left is None or right is None:
            return None
        return True

    elif isinstance(cond, Or):
        left  = _eval_condition_3vl(cond.left,  row)
        right = _eval_condition_3vl(cond.right, row)
        # TRUE OR anything = TRUE
        if left is True or right is True:
            return True
        # NULL OR FALSE = NULL,  NULL OR NULL = NULL
        if left is None or right is None:
            return None
        return False

    elif isinstance(cond, Not):
        child = _eval_condition_3vl(cond.child, row)
        # NOT NULL = NULL
        if child is None:
            return None
        return not child

    else:
        raise NotImplementedError(f"未知条件节点类型: {type(cond)}")


def _compare(left: Any, op: CmpOp, right: Any) -> bool:
    """
    执行单次比较。
    NULL 语义：任何涉及 None 的比较返回 False（和 SQL 行为一致）。
    """
    if left is None or right is None:
        return False

    if op == CmpOp.EQ:
        return left == right
    elif op == CmpOp.NEQ:
        return left != right
    elif op == CmpOp.GT:
        return left > right
    elif op == CmpOp.GTE:
        return left >= right
    elif op == CmpOp.LT:
        return left < right
    elif op == CmpOp.LTE:
        return left <= right
    else:
        raise NotImplementedError(f"未知比较运算符: {op}")


# ---------------------------------------------------------------------------
# 字段解析
# ---------------------------------------------------------------------------

def _resolve_field(field_name: str, row: Row) -> Any:
    """
    从 row 里取出字段值。

    查找策略（按优先级）：
      1. 精确匹配：row["o.amount"] → 直接返回
      2. 后缀匹配：field_name="amount"，row 里有 "o.amount" → 返回该值
         （用于 Project / Having 里写不带表前缀的列名）
      3. 找不到 → 返回 None，并打印警告
    """
    # 1. 精确匹配
    if field_name in row:
        return row[field_name]

    # 2. 后缀匹配（field_name 不含 "."）
    if "." not in field_name:
        matches = [v for k, v in row.items() if k.endswith(f".{field_name}")]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # 有歧义，说明两张表都有同名列，IR 应该用带前缀的字段名
            print(f"[python_ref] 警告：字段 {field_name!r} 有歧义，匹配到多列: "
                  f"{[k for k in row if k.endswith(f'.{field_name}')]}")
            return matches[0]

    print(f"[python_ref] 警告：字段 {field_name!r} 在当前行中找不到，row keys={list(row.keys())}")
    return None


# ---------------------------------------------------------------------------
# 直接运行此文件时：用手动构造的 IR 做快速验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from db.connector import create_tables, insert_rows, drop_tables, init_database
    from ir.nodes import pretty_print

    # 准备测试数据
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

    # ------------------------------------------------------------------
    # 测试一：单表 Filter + Project
    # SELECT user_id FROM orders WHERE amount > 60
    # 预期：[{"user_id": 80行对应user_id=1}, {"user_id": 200行对应user_id=2}]
    # ------------------------------------------------------------------
    print("\n=== 测试一：单表 Filter + Project ===")
    ir1 = Project(
        fields=["user_id"],
        child=Filter(
            condition=Compare("amount", CmpOp.GT, 60),
            child=Scan("orders")
        )
    )
    print(pretty_print(ir1))
    result1 = execute(ir1)
    print("结果:", result1)
    # 预期 user_id 出现 1（amount=80）和 2（amount=200）
    assert sorted(r["user_id"] for r in result1) == [1, 2], f"测试一失败: {result1}"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试二：GroupBy + Having + Project
    # SELECT user_id, SUM(amount) AS total FROM orders
    # GROUP BY user_id HAVING total > 100
    # 预期：user_id=1 total=130, user_id=2 total=200
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
    print(pretty_print(ir2))
    result2 = execute(ir2)
    print("结果:", result2)
    totals = {r["user_id"]: r["total"] for r in result2}
    assert totals == {1: 130.0, 2: 200.0}, f"测试二失败: {result2}"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试三：Join + Filter + Project
    # SELECT o.user_id, o.amount FROM orders o
    # INNER JOIN users u ON o.user_id = u.id
    # WHERE u.age > 18
    # 预期：user_id=1(age=25) 的订单 amount=50,80；user_id=3(age=30) 的订单 amount=30
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
    print(pretty_print(ir3))
    result3 = execute(ir3)
    print("结果:", result3)
    amounts = sorted(r["o.amount"] for r in result3)
    assert amounts == [30.0, 50.0, 80.0], f"测试三失败: {result3}"
    print("✓ 通过")

    # 清理
    print("\n=== 清理 ===")
    drop_tables(["orders", "users"])
    print("\n全部测试通过 ✓")
