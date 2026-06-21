"""
generator/ir_gen.py

随机 IR 生成器。

从一个 Schema 出发，随机生成一棵语义合法的 IR 树。
生成顺序固定：
  Scan → (可选) Join → (可选) Filter → (可选) GroupBy+Aggregate → (可选) Having → Project

合法性保证：
  - Join 的 on 条件字段来自两张表各自的列，且必须有外键关系
  - Having 只出现在 GroupBy 之后
  - 聚合函数只出现在 GroupBy 之后
  - Project 的字段来自当前查询范围内存在的列
  - 条件表达式的字段必须在当前上下文中存在
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ir.nodes import (
    Scan, Filter, Join, GroupBy, Having, Project,
    Compare, And, Or, Not, Aggregate,
    AggFunc, CmpOp, QueryNode, Condition,
)
from generator.schema_gen import Schema, TableSchema, ColType, Column, ForeignKey


# ---------------------------------------------------------------------------
# 生成上下文
# 记录当前 IR 树中"可见"的列和表信息
# ---------------------------------------------------------------------------

@dataclass
class GenContext:
    """
    IR 生成过程中的状态，随着节点逐层添加而更新。

    visible_cols : 当前查询上下文中可以引用的列，格式 "alias.colname"
    tables       : 当前涉及的表，{alias: TableSchema}
    agg_aliases  : 已定义的聚合别名列表（Having / Project 可以引用）
    group_fields : 当前 GroupBy 的分组字段列表
    has_groupby  : 是否已经有 GroupBy 节点
    """
    visible_cols : List[str]            = field(default_factory=list)
    tables       : dict                 = field(default_factory=dict)
    agg_aliases  : List[str]            = field(default_factory=list)
    group_fields : List[str]            = field(default_factory=list)
    has_groupby  : bool                 = False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def generate_ir(
    schema:       Schema,
    join_prob:    float = 0.4,   # 生成 Join 的概率
    filter_prob:  float = 0.6,   # 生成 Filter(WHERE) 的概率
    groupby_prob: float = 0.4,   # 生成 GroupBy 的概率
    having_prob:  float = 0.5,   # GroupBy 之后生成 Having 的概率
    seed:         Optional[int] = None,
) -> Tuple[QueryNode, GenContext]:
    """
    从 schema 随机生成一棵 IR 树。

    返回：(IR 树根节点, 生成上下文)
    上下文里记录了最终可见的列名，供调试和比较器使用。
    """
    if seed is not None:
        random.seed(seed)

    # ---------- Step 1: Scan ----------
    # 随机选一张主表
    main_table = random.choice(schema.tables)
    alias      = main_table.name[0]   # 用首字母作别名，如 orders → o
    node: QueryNode = Scan(main_table.name, alias)

    ctx = GenContext()
    _add_table_to_ctx(ctx, main_table, alias)

    # ---------- Step 2: Join（可选）----------
    # 只有 schema 里有外键关系，且随机决定加 Join 时才生成
    fk_pairs = _find_joinable_fks(schema, main_table)
    if fk_pairs and random.random() < join_prob:
        fk       = random.choice(fk_pairs)
        # 找到被 join 的表
        join_table = schema.get_table(
            fk.ref_table if fk.src_table == main_table.name else fk.src_table
        )
        join_alias = join_table.name[0]
        # 如果别名和主表冲突，加数字后缀
        if join_alias == alias:
            join_alias = join_alias + "2"

        on_cond = _make_fk_condition(fk, alias, join_alias, main_table, join_table)

        node = Join(
            left=Scan(main_table.name, alias),
            right=Scan(join_table.name, join_alias),
            on=on_cond,
        )
        _add_table_to_ctx(ctx, join_table, join_alias)

    # ---------- Step 3: Filter（可选，WHERE）----------
    if ctx.visible_cols and random.random() < filter_prob:
        cond = _generate_condition(ctx, schema)
        if cond is not None:
            node = Filter(condition=cond, child=node)

    # ---------- Step 4: GroupBy + Aggregate（可选）----------
    if ctx.visible_cols and random.random() < groupby_prob:
        # 随机选 1-2 个分组字段（只选非 VARCHAR 列，聚合数值更有意义）
        numeric_cols = _get_numeric_cols(ctx)
        all_cols     = ctx.visible_cols

        # 分组字段优先用全部可见列（任意类型都可以 group by）
        num_group = random.randint(1, min(2, len(all_cols)))
        group_fields = random.sample(all_cols, num_group)

        # 聚合字段：从数值列里选
        agg_cols = numeric_cols if numeric_cols else all_cols
        num_aggs = random.randint(1, min(2, len(agg_cols)))
        agg_list = []
        used_aliases = set()
        for _ in range(num_aggs):
            agg_col  = random.choice(agg_cols)
            agg_func = random.choice(list(AggFunc))
            # COUNT(*) 特殊处理
            if agg_func == AggFunc.COUNT:
                agg_field = "*"
            else:
                agg_field = agg_col
            # 生成唯一别名
            base_alias = f"{agg_func.value.lower()}_{agg_field.replace('.', '_').replace('*', 'all')}"
            alias_name = base_alias
            suffix = 1
            while alias_name in used_aliases:
                alias_name = f"{base_alias}_{suffix}"
                suffix += 1
            used_aliases.add(alias_name)
            agg_list.append(Aggregate(agg_func, agg_field, alias_name))

        node = GroupBy(fields=group_fields, aggregates=agg_list, child=node)

        ctx.has_groupby  = True
        ctx.group_fields = group_fields
        ctx.agg_aliases  = [a.alias for a in agg_list]

        # ---------- Step 5: Having（可选，需要 GroupBy）----------
        if ctx.agg_aliases and random.random() < having_prob:
            having_cond = _generate_having_condition(ctx)
            if having_cond is not None:
                node = Having(condition=having_cond, child=node)

    # ---------- Step 6: Project ----------
    # 决定最终输出哪些列
    project_fields = _choose_project_fields(ctx)
    if project_fields:
        node = Project(fields=project_fields, child=node)

    return node, ctx


# ---------------------------------------------------------------------------
# 上下文辅助
# ---------------------------------------------------------------------------

def _add_table_to_ctx(ctx: GenContext, table: TableSchema, alias: str) -> None:
    """把一张表的所有列加入 visible_cols，格式 alias.colname。"""
    ctx.tables[alias] = table
    for col in table.columns:
        ctx.visible_cols.append(f"{alias}.{col.name}")


def _get_numeric_cols(ctx: GenContext) -> List[str]:
    """返回上下文中所有数值类型（INT / FLOAT）的列名。"""
    result = []
    for alias, table in ctx.tables.items():
        for col in table.columns:
            if col.col_type in (ColType.INT, ColType.FLOAT):
                result.append(f"{alias}.{col.name}")
    return result


# ---------------------------------------------------------------------------
# 外键查找
# ---------------------------------------------------------------------------

def _find_joinable_fks(schema: Schema, main_table: TableSchema) -> List[ForeignKey]:
    """
    找出和 main_table 有直接外键关系的所有 FK 记录。
    包括：main_table 作为子表（src），或 main_table 作为父表（ref）。
    """
    result = []
    for fk in schema.fk_pairs():
        if fk.src_table == main_table.name or fk.ref_table == main_table.name:
            result.append(fk)
    return result


def _make_fk_condition(
    fk: ForeignKey,
    main_alias: str,
    join_alias: str,
    main_table: TableSchema,
    join_table: TableSchema,
) -> Compare:
    """
    根据外键关系构造 JOIN on 条件。
    例：orders.user_id = users.id
    →  Compare("o.user_id", CmpOp.EQ, "u.id")
    """
    if fk.src_table == main_table.name:
        # main_table 是子表，外键列在 main_table
        left  = f"{main_alias}.{fk.src_col}"
        right = f"{join_alias}.{fk.ref_col}"
    else:
        # main_table 是父表，外键列在 join_table
        left  = f"{join_alias}.{fk.src_col}"
        right = f"{main_alias}.{fk.ref_col}"
    return Compare(left, CmpOp.EQ, right)


# ---------------------------------------------------------------------------
# 条件生成
# ---------------------------------------------------------------------------

def _generate_condition(ctx: GenContext, schema: Schema, depth: int = 0) -> Optional[Condition]:
    """
    随机生成一个 WHERE 条件。
    depth 控制递归深度，避免生成过深的 AND/OR 树。
    """
    # 只用数值列做比较，避免字符串比较的复杂性
    numeric_cols = _get_numeric_cols(ctx)
    if not numeric_cols:
        return None

    # 深度 >= 2 时只生成叶节点（Compare）
    if depth >= 2:
        return _make_compare(numeric_cols)

    r = random.random()
    if r < 0.6:
        # 60%：简单比较
        return _make_compare(numeric_cols)
    elif r < 0.8:
        # 20%：AND
        left  = _generate_condition(ctx, schema, depth + 1)
        right = _generate_condition(ctx, schema, depth + 1)
        if left and right:
            return And(left, right)
        return left or right
    elif r < 0.95:
        # 15%：OR
        left  = _generate_condition(ctx, schema, depth + 1)
        right = _generate_condition(ctx, schema, depth + 1)
        if left and right:
            return Or(left, right)
        return left or right
    else:
        # 5%：NOT
        child = _make_compare(numeric_cols)
        return Not(child) if child else None


def _make_compare(numeric_cols: List[str]) -> Optional[Compare]:
    """生成一个简单的列和字面量的比较条件。"""
    if not numeric_cols:
        return None
    col = random.choice(numeric_cols)
    op  = random.choice([CmpOp.GT, CmpOp.GTE, CmpOp.LT, CmpOp.LTE, CmpOp.EQ])
    val = random.randint(1, 100)   # 字面量范围，和数据生成器保持一致
    return Compare(col, op, val)


def _generate_having_condition(ctx: GenContext) -> Optional[Condition]:
    """
    生成 HAVING 条件，只引用聚合别名，不引用原始列。
    """
    if not ctx.agg_aliases:
        return None
    alias_name = random.choice(ctx.agg_aliases)
    op  = random.choice([CmpOp.GT, CmpOp.GTE, CmpOp.LT, CmpOp.LTE])
    val = random.randint(1, 200)
    return Compare(alias_name, op, val)


# ---------------------------------------------------------------------------
# Project 字段选择
# ---------------------------------------------------------------------------

def _choose_project_fields(ctx: GenContext) -> List[str]:
    """
    选择 Project 输出哪些列。

    规则：
    - 有 GroupBy：只能选分组字段 + 聚合别名
    - 无 GroupBy：从 visible_cols 里随机选 1 到全部
    """
    if ctx.has_groupby:
        candidates = ctx.group_fields + ctx.agg_aliases
    else:
        candidates = ctx.visible_cols

    if not candidates:
        return []

    # 随机选 1 到全部列
    num = random.randint(1, len(candidates))
    return random.sample(candidates, num)


# ---------------------------------------------------------------------------
# 直接运行时：生成几棵 IR 树并打印
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from generator.schema_gen import generate_schema, print_schema
    from ir.nodes import pretty_print

    print("=== 示例一：seed=42，2 张表，有外键 ===")
    schema1 = generate_schema(num_tables=2, cols_per_table=3, fk_prob=1.0, seed=42)
    print_schema(schema1)
    print()
    for i in range(3):
        ir, ctx = generate_ir(schema1, seed=i)
        print(f"--- IR #{i+1} ---")
        print(pretty_print(ir))
        print(f"visible_cols : {ctx.visible_cols}")
        print(f"has_groupby  : {ctx.has_groupby}")
        print(f"agg_aliases  : {ctx.agg_aliases}")
        print()

    print("=== 示例二：seed=7，3 张表 ===")
    schema2 = generate_schema(num_tables=3, cols_per_table=2, fk_prob=0.8, seed=7)
    print_schema(schema2)
    print()
    ir, ctx = generate_ir(schema2, seed=99)
    print(pretty_print(ir))
    print(f"visible_cols : {ctx.visible_cols}")