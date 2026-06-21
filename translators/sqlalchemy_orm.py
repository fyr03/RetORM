"""
translators/sqlalchemy_orm.py

路径三：IR → SQLAlchemy ORM API。

工作方式：
  把 IR 树翻译成 SQLAlchemy Core 的查询调用链，通过 ORM 执行。
  这条路径是被测对象，它的结果和 python_ref / sql 路径做差分比较。

选择 SQLAlchemy Core 而不是 ORM Session：
  - Core 更接近 SQL 语义，和 IR 的映射更直接
  - ORM Session 需要先定义 Model 类，动态 schema 下更繁琐
  - Core 同样经过 SQLAlchemy 的查询编译、方言适配、类型映射流程，
    能覆盖我们关注的 ORM 层逻辑错误

使用 SQLAlchemy 的 reflect 机制：
  - 不需要手写 Model，直接从已有数据库表反射出 Table 对象
  - 这样 schema 变了不需要改翻译器代码
"""

import sys
from typing import List, Dict, Any, Optional

sys.path.insert(0, __file__.rsplit("/translators", 1)[0])

from sqlalchemy import (
    Table, MetaData, select, and_, or_, not_,
    func, asc, desc, text
)
from sqlalchemy.engine import Engine

from ir.nodes import (
    Scan, Filter, Join, GroupBy, Having, Project,
    Compare, And, Or, Not, Aggregate,
    AggFunc, CmpOp, QueryNode, Condition,
)
from db.connector import get_engine


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

Row  = Dict[str, Any]
Rows = List[Row]


# ---------------------------------------------------------------------------
# MetaData 缓存：同一个 engine 只 reflect 一次
# ---------------------------------------------------------------------------

_metadata: Optional[MetaData] = None


def _get_metadata(engine: Engine) -> MetaData:
    global _metadata
    if _metadata is None:
        _metadata = MetaData()
        _metadata.reflect(bind=engine)
    return _metadata


def reset_metadata() -> None:
    """
    schema 变化时（建表/删表后）调用，清除缓存，下次重新 reflect。
    在 tests/ 和 runner.py 里每次建完表后调用一次。
    """
    global _metadata
    _metadata = None


def _get_table(table_name: str, engine: Engine) -> Table:
    """根据表名返回 SQLAlchemy Table 对象。"""
    metadata = _get_metadata(engine)
    if table_name not in metadata.tables:
        raise KeyError(f"[sqlalchemy_orm] 表 '{table_name}' 不在 metadata 中，"
                       f"已知表: {list(metadata.tables.keys())}。"
                       f"请确认已建表并调用 reset_metadata()。")
    return metadata.tables[table_name]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def execute(ir: QueryNode) -> Rows:
    """
    接收一棵 IR 树，翻译成 SQLAlchemy 查询并执行，返回结果行列表。

    用法：
        from translators.sqlalchemy_orm import execute
        rows = execute(my_ir)
    """
    engine = get_engine()
    query, col_labels = _build_query(ir, engine)

    print(f"[sqlalchemy_orm] 生成 SQL:\n  {query}")

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = []
        for row in result:
            mapping = dict(row._mapping)
            rows.append(mapping)

    return rows


# ---------------------------------------------------------------------------
# 递归构建查询
# 返回值：(SQLAlchemy Select 对象, 列标签列表)
# col_labels 用于上层节点知道当前 SELECT 里有哪些列
# ---------------------------------------------------------------------------

def _build_query(node: QueryNode, engine: Engine):
    """
    递归处理 IR 树，返回 (select_obj, alias_map)。
    alias_map: {字段名或聚合别名 → SQLAlchemy 列表达式}
    """
    if isinstance(node, Scan):
        return _build_scan(node, engine)
    elif isinstance(node, Filter):
        return _build_filter(node, engine)
    elif isinstance(node, Join):
        return _build_join(node, engine)
    elif isinstance(node, GroupBy):
        return _build_groupby(node, engine)
    elif isinstance(node, Having):
        return _build_having(node, engine)
    elif isinstance(node, Project):
        return _build_project(node, engine)
    else:
        raise NotImplementedError(f"未知节点类型: {type(node)}")


# ---------------------------------------------------------------------------
# 构建过程中需要在节点间传递的上下文
# 用一个简单的 dict 携带：
#   "froms"    : List[Table or Join]  FROM 子句涉及的表
#   "tables"   : Dict[alias, Table]   alias → Table 对象的映射
#   "where"    : List[条件表达式]      WHERE 条件列表（AND 连接）
#   "groupby"  : List[列表达式]        GROUP BY 列表
#   "having"   : List[条件表达式]      HAVING 条件列表
#   "agg_map"  : Dict[alias, 列表达式] 聚合别名 → 聚合表达式
#   "select_cols": List[列表达式]      SELECT 列列表
# ---------------------------------------------------------------------------

def _build_scan(node: Scan, engine: Engine):
    tbl = _get_table(node.table, engine)
    # 用 alias 包装 Table，这样列引用会带表别名
    tbl_alias = tbl.alias(node.alias)

    ctx = {
        "froms":       [tbl_alias],
        "tables":      {node.alias: tbl_alias},
        "where":       [],
        "groupby":     [],
        "having":      [],
        "agg_map":     {},
        "select_cols": [tbl_alias],   # 默认 SELECT *（整个 alias 对象）
    }
    return ctx


def _build_filter(node: Filter, engine: Engine):
    ctx = _build_query(node.child, engine)
    cond = _build_condition(node.condition, ctx)
    ctx["where"].append(cond)
    return ctx


def _build_join(node: Join, engine: Engine):
    if not isinstance(node.left, Scan) or not isinstance(node.right, Scan):
        raise NotImplementedError("目前只支持 Scan JOIN Scan")

    left_ctx  = _build_scan(node.left,  engine)
    right_ctx = _build_scan(node.right, engine)

    left_tbl  = left_ctx["tables"][node.left.alias]
    right_tbl = right_ctx["tables"][node.right.alias]

    # 合并上下文
    ctx = {
        "froms":       left_ctx["froms"],
        "tables":      {**left_ctx["tables"], **right_ctx["tables"]},
        "where":       [],
        "groupby":     [],
        "having":      [],
        "agg_map":     {},
        "select_cols": [left_tbl, right_tbl],  # 默认两张表都 SELECT *
    }

    # 构建 ON 条件
    on_cond = _build_condition(node.on, ctx)

    # 把 JOIN 合并进 froms：用 SQLAlchemy 的 join()
    joined = left_tbl.join(right_tbl, on_cond)
    ctx["froms"] = [joined]

    return ctx


def _build_groupby(node: GroupBy, engine: Engine):
    ctx = _build_query(node.child, engine)

    # 分组字段
    groupby_cols = [_resolve_col(f, ctx) for f in node.fields]
    ctx["groupby"] = groupby_cols

    # 聚合表达式
    agg_exprs = []
    for agg in node.aggregates:
        expr = _build_aggregate(agg, ctx)
        labeled = expr.label(agg.alias)
        ctx["agg_map"][agg.alias] = labeled
        agg_exprs.append(labeled)

    # SELECT = 分组字段 + 聚合表达式
    ctx["select_cols"] = groupby_cols + agg_exprs

    return ctx


def _build_having(node: Having, engine: Engine):
    ctx = _build_query(node.child, engine)  # child 必须是 GroupBy
    cond = _build_condition(node.condition, ctx)
    ctx["having"].append(cond)
    return ctx


def _build_project(node: Project, engine: Engine):
    ctx = _build_query(node.child, engine)

    cols = []
    for f in node.fields:
        if f in ctx.get("agg_map", {}):
            cols.append(ctx["agg_map"][f])
        else:
            cols.append(_resolve_col(f, ctx))

    ctx["select_cols"] = cols
    return ctx


# ---------------------------------------------------------------------------
# 从 ctx 组装最终 Select 对象
# _build_query 返回的是 ctx dict，execute() 调用此函数最终生成查询
# ---------------------------------------------------------------------------

def _build_query(node: QueryNode, engine: Engine):
    """重写为两步：先收集 ctx，再组装 Select。"""
    ctx = _collect_ctx(node, engine)
    stmt = _assemble(ctx)
    return stmt, ctx


def _collect_ctx(node: QueryNode, engine: Engine) -> dict:
    if isinstance(node, Scan):
        return _build_scan(node, engine)
    elif isinstance(node, Filter):
        return _build_filter_ctx(node, engine)
    elif isinstance(node, Join):
        return _build_join_ctx(node, engine)
    elif isinstance(node, GroupBy):
        return _build_groupby_ctx(node, engine)
    elif isinstance(node, Having):
        return _build_having_ctx(node, engine)
    elif isinstance(node, Project):
        return _build_project_ctx(node, engine)
    else:
        raise NotImplementedError(f"未知节点类型: {type(node)}")


def _build_filter_ctx(node: Filter, engine: Engine):
    ctx = _collect_ctx(node.child, engine)
    cond = _build_condition(node.condition, ctx)
    ctx["where"].append(cond)
    return ctx


def _build_join_ctx(node: Join, engine: Engine):
    if not isinstance(node.left, Scan) or not isinstance(node.right, Scan):
        raise NotImplementedError("目前只支持 Scan JOIN Scan")

    left_ctx  = _build_scan(node.left,  engine)
    right_ctx = _build_scan(node.right, engine)

    left_tbl  = left_ctx["tables"][node.left.alias]
    right_tbl = right_ctx["tables"][node.right.alias]

    ctx = {
        "froms":       left_ctx["froms"],
        "tables":      {**left_ctx["tables"], **right_ctx["tables"]},
        "where":       [],
        "groupby":     [],
        "having":      [],
        "agg_map":     {},
        "select_cols": [left_tbl, right_tbl],
    }

    on_cond = _build_condition(node.on, ctx)
    joined  = left_tbl.join(right_tbl, on_cond)
    ctx["froms"] = [joined]
    return ctx


def _build_groupby_ctx(node: GroupBy, engine: Engine):
    ctx = _collect_ctx(node.child, engine)

    groupby_cols = [_resolve_col(f, ctx) for f in node.fields]
    ctx["groupby"] = groupby_cols

    agg_exprs = []
    for agg in node.aggregates:
        expr    = _build_aggregate(agg, ctx)
        labeled = expr.label(agg.alias)
        ctx["agg_map"][agg.alias] = labeled
        agg_exprs.append(labeled)

    ctx["select_cols"] = groupby_cols + agg_exprs
    return ctx


def _build_having_ctx(node: Having, engine: Engine):
    ctx = _collect_ctx(node.child, engine)
    cond = _build_condition(node.condition, ctx)
    ctx["having"].append(cond)
    return ctx


def _build_project_ctx(node: Project, engine: Engine):
    ctx = _collect_ctx(node.child, engine)
    cols = []
    for f in node.fields:
        if f in ctx.get("agg_map", {}):
            cols.append(ctx["agg_map"][f])
        else:
            col = _resolve_col(f, ctx)
            # 用 IR 字段名作为 label，避免 SQLAlchemy 对重复列名自动改名（如 id → id_1）
            # label 里 "." 不合法，替换成 "_"
            label_name = f.replace(".", "_")
            cols.append(col.label(label_name))
    ctx["select_cols"] = cols
    return ctx


def _assemble(ctx: dict):
    """
    把 ctx 组装成 SQLAlchemy Select 语句。
    """
    select_cols = ctx["select_cols"]

    # select_cols 里可能有整个 Table alias 对象（SELECT *），展开成列列表
    expanded = []
    for item in select_cols:
        if hasattr(item, "c"):
            # Table 或 AliasedTable：展开所有列
            expanded.extend(item.c)
        else:
            expanded.append(item)

    stmt = select(*expanded)

    # FROM / JOIN
    for from_clause in ctx["froms"]:
        stmt = stmt.select_from(from_clause)

    # WHERE
    if ctx["where"]:
        stmt = stmt.where(and_(*ctx["where"]))

    # GROUP BY
    if ctx["groupby"]:
        stmt = stmt.group_by(*ctx["groupby"])

    # HAVING
    if ctx["having"]:
        stmt = stmt.having(and_(*ctx["having"]))

    return stmt


# ---------------------------------------------------------------------------
# 条件构建
# ---------------------------------------------------------------------------

def _build_condition(cond: Condition, ctx: dict):
    """把条件节点翻译成 SQLAlchemy 条件表达式。"""

    if isinstance(cond, Compare):
        left_col = _resolve_col(cond.field, ctx)
        op       = cond.op

        # 右值：字符串且含 "." → 列引用；否则字面量
        if isinstance(cond.value, str) and "." in cond.value:
            right = _resolve_col(cond.value, ctx)
        else:
            right = cond.value

        if op == CmpOp.EQ:
            return left_col == right
        elif op == CmpOp.NEQ:
            return left_col != right
        elif op == CmpOp.GT:
            return left_col > right
        elif op == CmpOp.GTE:
            return left_col >= right
        elif op == CmpOp.LT:
            return left_col < right
        elif op == CmpOp.LTE:
            return left_col <= right
        else:
            raise NotImplementedError(f"未知比较运算符: {op}")

    elif isinstance(cond, And):
        return and_(
            _build_condition(cond.left,  ctx),
            _build_condition(cond.right, ctx),
        )

    elif isinstance(cond, Or):
        return or_(
            _build_condition(cond.left,  ctx),
            _build_condition(cond.right, ctx),
        )

    elif isinstance(cond, Not):
        return not_(_build_condition(cond.child, ctx))

    else:
        raise NotImplementedError(f"未知条件节点类型: {type(cond)}")


# ---------------------------------------------------------------------------
# 聚合函数构建
# ---------------------------------------------------------------------------

def _build_aggregate(agg: Aggregate, ctx: dict):
    """把 Aggregate 节点翻译成 SQLAlchemy 聚合函数表达式。"""
    if agg.func == AggFunc.COUNT:
        if agg.field == "*":
            return func.count()
        else:
            col = _resolve_col(agg.field, ctx)
            return func.count(col)
    else:
        col = _resolve_col(agg.field, ctx)
        fn_map = {
            AggFunc.SUM: func.sum,
            AggFunc.AVG: func.avg,
            AggFunc.MAX: func.max,
            AggFunc.MIN: func.min,
        }
        return fn_map[agg.func](col)


# ---------------------------------------------------------------------------
# 列解析
# ---------------------------------------------------------------------------

def _resolve_col(field_name: str, ctx: dict):
    """
    把字段名解析成 SQLAlchemy 列表达式。

    支持格式：
      "o.amount"  → ctx["tables"]["o"].c.amount
      "amount"    → 在所有表里搜索 amount 列（唯一时直接返回）
      "total"     → 聚合别名，从 ctx["agg_map"] 取
    """
    # 1. 先查聚合别名
    if field_name in ctx.get("agg_map", {}):
        return ctx["agg_map"][field_name]

    # 2. 带表前缀
    if "." in field_name:
        table_alias, col_name = field_name.split(".", 1)
        tbl = ctx["tables"].get(table_alias)
        if tbl is None:
            raise KeyError(f"[sqlalchemy_orm] 找不到表别名 '{table_alias}'，"
                           f"已知别名: {list(ctx['tables'].keys())}")
        return tbl.c[col_name]

    # 3. 不带前缀：在所有表里搜索
    matches = []
    for tbl in ctx["tables"].values():
        if field_name in tbl.c:
            matches.append(tbl.c[field_name])

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"[sqlalchemy_orm] 警告：字段 '{field_name}' 有歧义，"
              f"匹配到多列，取第一个")
        return matches[0]
    else:
        raise KeyError(f"[sqlalchemy_orm] 找不到字段 '{field_name}'，"
                       f"已知表: {list(ctx['tables'].keys())}")


# ---------------------------------------------------------------------------
# 直接运行此文件时：三个测试用例验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from db.connector import (
        create_tables, insert_rows, drop_tables,
        init_database, dispose_engine,
    )
    from ir.nodes import pretty_print

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

    # reflect 之前先清除缓存
    reset_metadata()

    # ------------------------------------------------------------------
    # 测试一：单表 Filter + Project
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
    totals = {r["user_id"]: float(r["total"]) for r in result2}
    assert totals == {1: 130.0, 2: 200.0}, f"测试二失败: {result2}"
    print("✓ 通过")

    # ------------------------------------------------------------------
    # 测试三：Join + Filter + Project
    # 预期：amount = 30, 50, 80
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
    amounts = sorted(float(r["amount"]) for r in result3)
    assert amounts == [30.0, 50.0, 80.0], f"测试三失败: {result3}"
    print("✓ 通过")

    # 清理
    print("\n=== 清理 ===")
    drop_tables(["orders", "users"])
    dispose_engine()
    print("\n全部测试通过 ✓")