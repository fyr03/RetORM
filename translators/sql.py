"""
translators/sql.py

路径二：IR → Raw SQL。

工作方式：
  把 IR 树机械地翻译成一条 SQL 字符串，直接在 MySQL 上执行。
  不做任何优化，就是节点到 SQL 子句的直接映射。

翻译规则：
  Scan        → FROM table AS alias
  Filter      → WHERE ...
  Join        → INNER JOIN ... ON ...
  GroupBy     → GROUP BY ...
  Having      → HAVING ...
  Project     → SELECT col1, col2, ...
  （无 Project）→ SELECT *

列引用约定（和 python_ref 保持一致）：
  右值是字符串且含 "." → 列引用，不加引号
  其他字符串           → 字符串字面量，加单引号
"""

import sys
from typing import List, Tuple, Optional

sys.path.insert(0, __file__.rsplit("/translators", 1)[0])

from ir.nodes import (
    Scan, Filter, Join, GroupBy, Having, Project,
    Compare, And, Or, Not, Aggregate,
    AggFunc, CmpOp, QueryNode, Condition,
)
from db.connector import execute_sql


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def execute(ir: QueryNode) -> List[dict]:
    """
    接收一棵 IR 树，翻译成 SQL 并执行，返回结果行列表。

    用法：
        from translators.sql import execute
        rows = execute(my_ir)
    """
    sql = translate(ir)
    print(f"[sql] 生成 SQL:\n  {sql}")
    rows = execute_sql(sql)
    return rows


def translate(ir: QueryNode) -> str:
    """
    只翻译，不执行，返回 SQL 字符串。
    方便调试时单独查看生成的 SQL。
    """
    select_clause, from_clause, where_clause, \
    groupby_clause, having_clause = _translate(ir)

    # 拼装完整 SQL
    parts = [select_clause, from_clause]
    if where_clause:
        parts.append(f"WHERE {where_clause}")
    if groupby_clause:
        parts.append(f"GROUP BY {groupby_clause}")
    if having_clause:
        parts.append(f"HAVING {having_clause}")

    return "\n".join(parts) + ";"


# ---------------------------------------------------------------------------
# 递归翻译
# 返回值是一个五元组：
#   (select_clause, from_clause, where_clause, groupby_clause, having_clause)
# 各子句都是字符串，空子句用 None 表示
# ---------------------------------------------------------------------------

def _translate(node: QueryNode) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:

    if isinstance(node, Scan):
        return _translate_scan(node)

    elif isinstance(node, Filter):
        return _translate_filter(node)

    elif isinstance(node, Join):
        return _translate_join(node)

    elif isinstance(node, GroupBy):
        return _translate_groupby(node)

    elif isinstance(node, Having):
        return _translate_having(node)

    elif isinstance(node, Project):
        return _translate_project(node)

    else:
        raise NotImplementedError(f"未知节点类型: {type(node)}")


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _translate_scan(node: Scan):
    """
    SELECT * FROM `table` AS `alias`
    """
    select  = "SELECT *"
    from_   = f"FROM `{node.table}` AS `{node.alias}`"
    return select, from_, None, None, None


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def _translate_filter(node: Filter):
    """
    在子查询基础上追加 WHERE 子句。
    如果子查询已经有 WHERE（理论上不应出现，IR 生成器保证），则用 AND 合并。
    """
    select, from_, where, groupby, having = _translate(node.child)
    new_where = _translate_condition(node.condition)

    if where:
        # 保险起见，合并已有的 WHERE
        combined = f"({where}) AND ({new_where})"
    else:
        combined = new_where

    return select, from_, combined, groupby, having


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def _translate_join(node: Join):
    """
    FROM `left_table` AS `left_alias`
    INNER JOIN `right_table` AS `right_alias`
    ON left_col = right_col

    目前只支持两侧都是 Scan 的简单 join。
    """
    if not isinstance(node.left, Scan) or not isinstance(node.right, Scan):
        raise NotImplementedError("目前只支持 Scan JOIN Scan，不支持嵌套 JOIN")

    left  = node.left
    right = node.right
    on    = _translate_condition(node.on)

    select = "SELECT *"
    from_  = (
        f"FROM `{left.table}` AS `{left.alias}`\n"
        f"INNER JOIN `{right.table}` AS `{right.alias}` ON {on}"
    )
    return select, from_, None, None, None


# ---------------------------------------------------------------------------
# GroupBy
# ---------------------------------------------------------------------------

def _translate_groupby(node: GroupBy):
    """
    在子查询基础上追加 GROUP BY 子句，同时把 SELECT 改成包含聚合函数。
    """
    select, from_, where, _, having = _translate(node.child)

    # 分组字段
    group_cols  = ", ".join(_quote_field(f) for f in node.fields)

    # SELECT 部分：分组字段 + 聚合表达式
    agg_exprs   = [_translate_aggregate(a) for a in node.aggregates]
    select_cols = ", ".join(
        [_quote_field(f) for f in node.fields] + agg_exprs
    )
    select = f"SELECT {select_cols}"

    return select, from_, where, group_cols, having


def _translate_aggregate(agg: Aggregate) -> str:
    """
    把 Aggregate 节点翻译成 SQL 聚合表达式。
    例：Aggregate(SUM, "o.amount", "total") → SUM(`o`.`amount`) AS `total`
    """
    if agg.field == "*":
        expr = f"{agg.func.value}(*)"
    else:
        expr = f"{agg.func.value}({_quote_field(agg.field)})"
    return f"{expr} AS `{agg.alias}`"


# ---------------------------------------------------------------------------
# Having
# ---------------------------------------------------------------------------

def _translate_having(node: Having):
    """
    在子查询（必须是 GroupBy）基础上追加 HAVING 子句。

    MySQL 不允许在 HAVING 里引用聚合别名，必须展开成完整聚合表达式。
    例：HAVING `count_all` < 126
        → HAVING COUNT(*) < 126
    """
    select, from_, where, groupby, _ = _translate(node.child)
    # 收集聚合别名 → 完整表达式的映射（不含 AS alias 部分）
    agg_expr_map = _collect_agg_exprs(node.child)
    having = _translate_condition(node.condition, agg_expr_map=agg_expr_map)
    return select, from_, where, groupby, having


def _collect_agg_exprs(node) -> dict:
    """
    遍历子树，收集 {alias: "COUNT(*)"} 形式的映射。
    注意：这里只要聚合表达式本身，不含 AS alias。
    """
    result = {}
    if isinstance(node, GroupBy):
        for agg in node.aggregates:
            if agg.field == "*":
                expr = f"{agg.func.value}(*)"
            else:
                expr = f"{agg.func.value}({_quote_field(agg.field)})"
            result[agg.alias] = expr
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Having):
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Filter):
        result.update(_collect_agg_exprs(node.child))
    elif isinstance(node, Join):
        result.update(_collect_agg_exprs(node.left))
        result.update(_collect_agg_exprs(node.right))
    return result


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

def _translate_project(node: Project):
    """
    把 SELECT * 替换成 SELECT 指定列。

    关键点：如果子节点是 GroupBy（或外包 Having 的 GroupBy），
    Project 的字段里可能包含聚合别名（如 "total"）。
    MySQL 不允许在 SELECT 里直接写别名，
    必须把别名替换回完整的聚合表达式（如 SUM(`amount`) AS `total`）。
    """
    select, from_, where, groupby, having = _translate(node.child)

    # 收集子树中所有聚合节点，建立 alias → 聚合表达式 的映射
    agg_map = _collect_aggregates(node.child)

    col_exprs = []
    for f in node.fields:
        if f in agg_map:
            # 聚合别名 → 展开成完整聚合表达式
            col_exprs.append(agg_map[f])
        else:
            col_exprs.append(_quote_field(f))

    select = f"SELECT {', '.join(col_exprs)}"
    return select, from_, where, groupby, having


def _collect_aggregates(node: QueryNode) -> dict:
    """
    遍历 IR 子树，收集所有 Aggregate 节点，
    返回 {alias: "SUM(`field`) AS `alias`"} 的映射。
    """
    result = {}
    if isinstance(node, GroupBy):
        for agg in node.aggregates:
            result[agg.alias] = _translate_aggregate(agg)
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Having):
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Filter):
        result.update(_collect_aggregates(node.child))
    elif isinstance(node, Join):
        result.update(_collect_aggregates(node.left))
        result.update(_collect_aggregates(node.right))
    # Scan 是叶节点，无聚合
    return result


# ---------------------------------------------------------------------------
# 条件翻译
# ---------------------------------------------------------------------------

def _translate_condition(cond: Condition, agg_expr_map: dict = None) -> str:
    """
    把条件节点翻译成 SQL 条件字符串。

    agg_expr_map: {alias: "COUNT(*)"} 映射，用于 HAVING 里展开聚合别名。
                  WHERE 条件不需要传这个参数。
    """
    agg_expr_map = agg_expr_map or {}

    if isinstance(cond, Compare):
        field_name = cond.field
        # 如果字段名是聚合别名，展开成完整聚合表达式
        if field_name in agg_expr_map:
            left = agg_expr_map[field_name]
        else:
            left = _quote_field(field_name)
        op    = cond.op.value
        right = _quote_value(cond.value)
        return f"{left} {op} {right}"

    elif isinstance(cond, And):
        l = _translate_condition(cond.left,  agg_expr_map)
        r = _translate_condition(cond.right, agg_expr_map)
        return f"({l}) AND ({r})"

    elif isinstance(cond, Or):
        l = _translate_condition(cond.left,  agg_expr_map)
        r = _translate_condition(cond.right, agg_expr_map)
        return f"({l}) OR ({r})"

    elif isinstance(cond, Not):
        c = _translate_condition(cond.child, agg_expr_map)
        return f"NOT ({c})"

    else:
        raise NotImplementedError(f"未知条件节点类型: {type(cond)}")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _quote_field(field_name: str) -> str:
    """
    把字段名转成 SQL 里的带反引号格式。

    "amount"      → `amount`
    "o.amount"    → `o`.`amount`
    "total"       → `total`      （聚合别名，无前缀）
    """
    if "." in field_name:
        parts = field_name.split(".", 1)
        return f"`{parts[0]}`.`{parts[1]}`"
    else:
        return f"`{field_name}`"


def _quote_value(value) -> str:
    """
    把右值转成 SQL 字面量。

    约定：字符串且含 "." → 列引用，用 _quote_field 处理
          None            → NULL
          str             → 'value'（加单引号）
          int / float     → 直接转字符串
    """
    if value is None:
        return "NULL"
    elif isinstance(value, str) and "." in value:
        # 列引用
        return _quote_field(value)
    elif isinstance(value, str):
        # 字符串字面量，转义单引号
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        # bool 要在 int 之前判断，Python 里 bool 是 int 的子类
        return "1" if value else "0"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return f"'{value}'"


# ---------------------------------------------------------------------------
# 直接运行此文件时：用和 python_ref 相同的三个测试用例验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from db.connector import create_tables, insert_rows, drop_tables, init_database
    from ir.nodes import pretty_print, Aggregate, AggFunc, CmpOp

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
    # SELECT `user_id` FROM `orders` AS `orders` WHERE `amount` > 60
    # 预期：user_id = 1, 2
    # ------------------------------------------------------------------
    print("\n=== 测试一：单表 Filter + Project ===")
    ir1 = Project(
        fields=["user_id"],
        child=Filter(
            condition=Compare("amount", CmpOp.GT, 60),
            child=Scan("orders")
        )
    )
    result1 = execute(ir1)
    print("结果:", result1)
    assert sorted(r["user_id"] for r in result1) == [1, 2], f"测试一失败: {result1}"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试二：GroupBy + Having + Project
    # SELECT `user_id`, SUM(`amount`) AS `total`
    # FROM `orders` AS `orders`
    # GROUP BY `user_id`
    # HAVING `total` > 100
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
    result2 = execute(ir2)
    print("结果:", result2)
    totals = {r["user_id"]: r["total"] for r in result2}
    assert totals == {1: 130.0, 2: 200.0}, f"测试二失败: {result2}"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试三：Join + Filter + Project
    # SELECT `o`.`user_id`, `o`.`amount`
    # FROM `orders` AS `o`
    # INNER JOIN `users` AS `u` ON `o`.`user_id` = `u`.`id`
    # WHERE `u`.`age` > 18
    # 预期：amount = 50, 80（user1,age25）; 30（user3,age30）
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
    result3 = execute(ir3)
    print("结果:", result3)
    # SQL 路径返回的列名是 MySQL 原始列名（不带表前缀），
    # 这里取每行第二个值（amount）做断言，兼容列名差异
    amounts = sorted(list(r.values())[1] for r in result3)
    assert amounts == [30.0, 50.0, 80.0], f"测试三失败: {result3}"
    print("✓ 通过")

    # 清理
    print("\n=== 清理 ===")
    drop_tables(["orders", "users"])
    print("\n全部测试通过 ✓")