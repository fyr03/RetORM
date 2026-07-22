"""
generator/data_gen.py

测试数据生成器。

两种策略：
  1. 随机生成（默认）：为每张表生成若干随机行，简单快速
  2. Z3 约束求解（可选）：根据 IR 的 Filter / Having 条件反推满足条件的记录，
     保证查询返回非空结果，提高差分测试的有效性

当前实现：
  - 随机生成已完整实现
  - Z3 部分实现了基础框架，支持简单的 Compare 条件（AND/OR/NOT 递归展开）
  - Z3 超时或不可用时自动回退到随机生成

数据值域约定（和 ir_gen.py 的条件字面量范围对齐）：
  INT    : 1 ~ 100
  FLOAT  : 1.0 ~ 100.0（保留 1 位小数）
  VARCHAR: 随机短字符串
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import string
from typing import List, Dict, Any, Optional, Tuple

from ir.nodes import (
    Scan, Filter, Join, GroupBy, Having, Project,
    Compare, And, Or, Not, Aggregate,
    AggFunc, CmpOp, QueryNode, Condition,
)
from generator.schema_gen import Schema, TableSchema, ColType, Column


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

Row     = Dict[str, Any]
TableData = Dict[str, List[Row]]   # table_name → rows


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def generate_and_insert(
    schema:     Schema,
    ir:         QueryNode,
    rows_per_table: int = 8,
    use_z3:     bool = True,
    z3_timeout: int  = 5,
    stress_mode: str = "balanced",
    seed:       Optional[int] = None,
) -> TableData:
    """
    为 schema 里的所有表生成测试数据并插入数据库。

    参数：
        schema         : 当前测试用的 Schema
        ir             : 当前 IR 树（用于 Z3 约束提取）
        rows_per_table : 每张表生成多少行
        use_z3         : 是否尝试 Z3 约束求解
        z3_timeout     : Z3 超时时间（秒）
        seed           : 随机种子

    返回：
        TableData：{table_name: [row_dict, ...]}，方便调试查看
    """
    if seed is not None:
        random.seed(seed)

    # 先尝试 Z3（只有安装了 z3-solver 且 use_z3=True 时才用）
    if use_z3:
        try:
            data = _generate_with_z3(
                schema,
                ir,
                rows_per_table,
                z3_timeout,
                stress_mode=stress_mode,
            )
            if data:
                _insert_all(schema, data)
                return data
        except Exception as e:
            print(f"[data_gen] Z3 生成失败（{e}），回退到随机生成")

    # 随机生成
    data = _generate_random(schema, rows_per_table, stress_mode=stress_mode)
    _insert_all(schema, data)
    return data


def generate_random_only(
    schema:         Schema,
    rows_per_table: int = 8,
    stress_mode:    str = "balanced",
    seed:           Optional[int] = None,
) -> TableData:
    """
    只用随机生成，不尝试 Z3。
    快速调试时使用。
    """
    if seed is not None:
        random.seed(seed)
    data = _generate_random(schema, rows_per_table, stress_mode=stress_mode)
    _insert_all(schema, data)
    return data


# ---------------------------------------------------------------------------
# 随机数据生成
# ---------------------------------------------------------------------------

def _generate_random(
    schema: Schema,
    rows_per_table: int,
    stress_mode: str = "balanced",
) -> TableData:
    """
    为 schema 里每张表随机生成数据。

    外键处理：
      如果表 B 有外键指向表 A，B 的外键列值从 A 已生成的 id 中随机选取，
      保证外键值合法（不会出现悬空引用）。
    """
    data: TableData = {}

    # 按拓扑顺序生成（父表先生成，子表后生成）
    from generator.schema_gen import generate_drop_sqls
    ordered_names = list(reversed(generate_drop_sqls(schema)))  # drop 是逆序，reverse 回来就是建表顺序

    for tname in ordered_names:
        table  = schema.get_table(tname)
        rows = _generate_table_rows(table, rows_per_table, data, stress_mode=stress_mode)
        data[tname] = rows

    return data


def _generate_table_rows(
    table:  TableSchema,
    n:      int,
    existing_data: TableData,
    stress_mode: str = "balanced",
) -> List[Row]:
    """为单张表生成 n 行数据。"""
    rows = []
    value_pools = {
        col.name: _build_value_pool(col, stress_mode)
        for col in table.columns
        if not col.is_pk and not _is_fk_col(col.name, table)
    }
    for i in range(n):
        row: Row = {}
        for col in table.columns:
            if col.is_pk:
                row[col.name] = i + 1   # id 从 1 开始，保证唯一
            elif _is_fk_col(col.name, table):
                # 外键列：从父表的 id 里随机选
                ref_table_name = _get_fk_ref_table(col.name, table)
                if ref_table_name and ref_table_name in existing_data:
                    parent_ids = [r["id"] for r in existing_data[ref_table_name]]
                    row[col.name] = (
                        _choose_parent_id(parent_ids, stress_mode) if parent_ids else 1
                    )
                else:
                    row[col.name] = random.randint(1, n)
            else:
                row[col.name] = _random_value(
                    col,
                    stress_mode=stress_mode,
                    pool=value_pools.get(col.name),
                )
        rows.append(row)
    return rows


def _random_value(
    col: Column,
    stress_mode: str = "balanced",
    pool: Optional[List[Any]] = None,
) -> Any:
    """根据列类型生成随机值。"""
    # 以 nullable_prob 决定是否生成 NULL
    null_prob = 0.15
    if stress_mode == "null_heavy":
        null_prob = 0.45
    elif stress_mode == "groupby_heavy":
        null_prob = 0.22
    if col.nullable and random.random() < null_prob:
        return None

    if pool:
        reuse_prob = 0.55
        if stress_mode in ("groupby_heavy", "join_heavy", "duplicate_column_heavy"):
            reuse_prob = 0.8
        elif stress_mode == "null_heavy":
            reuse_prob = 0.7
        if random.random() < reuse_prob:
            return random.choice(pool)

    return _random_scalar_value(col)


def _random_scalar_value(col: Column) -> Any:
    """Generate a non-NULL scalar value for a single column."""
    if col.col_type == ColType.INT:
        return random.randint(1, 100)
    if col.col_type == ColType.FLOAT:
        return round(random.uniform(1.0, 100.0), 1)
    if col.col_type == ColType.VARCHAR:
        length = random.randint(3, 8)
        return "".join(random.choices(string.ascii_lowercase, k=length))
    return random.randint(1, 100)


def _build_value_pool(col: Column, stress_mode: str) -> List[Any]:
    """Build a small reuse pool to increase duplicate and boundary-value coverage."""
    pool_size = 2
    if stress_mode in ("groupby_heavy", "join_heavy", "duplicate_column_heavy"):
        pool_size = 3
    return [_random_scalar_value(col) for _ in range(pool_size)]


def _choose_parent_id(parent_ids: List[int], stress_mode: str) -> int:
    """Bias FK selection toward collisions in stress modes that benefit from denser joins."""
    if not parent_ids:
        return 1

    if stress_mode in ("join_heavy", "groupby_heavy", "duplicate_column_heavy"):
        hot_ids = parent_ids[: min(2, len(parent_ids))]
        if hot_ids and random.random() < 0.8:
            return random.choice(hot_ids)

    return random.choice(parent_ids)


def _is_fk_col(col_name: str, table: TableSchema) -> bool:
    """判断列名是否是外键列。"""
    return any(fk.src_col == col_name for fk in table.fks)


def _get_fk_ref_table(col_name: str, table: TableSchema) -> Optional[str]:
    """根据外键列名找到它引用的表名。"""
    for fk in table.fks:
        if fk.src_col == col_name:
            return fk.ref_table
    return None


# ---------------------------------------------------------------------------
# Z3 约束求解
# ---------------------------------------------------------------------------

def _generate_with_z3(
    schema:     Schema,
    ir:         QueryNode,
    rows_per_table: int,
    timeout_sec: int,
    stress_mode: str = "balanced",
) -> Optional[TableData]:
    """
    用 Z3 根据 IR 中“简单可约束化”的条件反推一小部分满足条件的记录。

    策略：
      1. 从 IR 树中提取简单数值 Filter 条件，以及 Join on 等值条件
      2. 为每个表别名的“第一行”建立 Z3 变量
      3. 添加可解析的约束
      4. 求解，把解转成 Python 数据
      5. 其余行用随机值填充

    注意：
      - 当前不是完整的多行/完整 SQL 语义求解器
      - Having 里引用聚合别名的复杂约束仍然跳过
      - 目标是提高有效测试比例，而不是保证所有复杂查询都非空
    """
    import z3

    # 提取 IR 中可直接转成 Z3 的简单约束
    alias_map = _extract_alias_map(ir)
    constraints = _extract_z3_constraints(ir)
    if not constraints:
        return None   # 没有可利用的条件，让调用方回退随机

    # 确定涉及的表别名
    if not alias_map:
        return None

    data: TableData = {}
    solver = z3.Solver()
    solver.set("timeout", timeout_sec * 1000)  # Z3 timeout 单位是毫秒

    # 为每个表别名的第一行建立 Z3 变量
    # 格式：z3_vars[alias][col_name] = z3 变量
    z3_vars: Dict[str, Dict[str, Any]] = {}
    table_primary_alias: Dict[str, str] = {}

    for alias, tname in alias_map.items():
        table = schema.get_table(tname)
        if table is None:
            continue
        z3_vars[alias] = {}
        table_primary_alias.setdefault(tname, alias)
        for col in table.non_pk_columns():
            if col.col_type in (ColType.INT, ColType.FLOAT):
                var = z3.Real(f"{alias}_{col.name}_0")
                z3_vars[alias][col.name] = var
                # 值域约束
                solver.add(var >= 1)
                solver.add(var <= 100)

    # 把可解析条件加入 solver
    for cond in constraints:
        z3_cond = _condition_to_z3(cond, z3_vars, alias_map)
        if z3_cond is not None:
            solver.add(z3_cond)

    # 求解
    result = solver.check()
    if result != z3.sat:
        print(f"[data_gen] Z3 无解（result={result}），回退随机生成")
        return None

    model = solver.model()

    # 从拓扑顺序生成数据
    from generator.schema_gen import generate_drop_sqls
    ordered_names = list(reversed(generate_drop_sqls(schema)))

    for tname in ordered_names:
        table = schema.get_table(tname)
        rows = []
        value_pools = {
            col.name: _build_value_pool(col, stress_mode)
            for col in table.columns
            if not col.is_pk and not _is_fk_col(col.name, table)
        }

        for i in range(rows_per_table):
            row: Row = {}
            for col in table.columns:
                if col.is_pk:
                    row[col.name] = i + 1
                elif _is_fk_col(col.name, table):
                    ref_tname = _get_fk_ref_table(col.name, table)
                    if ref_tname and ref_tname in data:
                        parent_ids = [r["id"] for r in data[ref_tname]]
                        row[col.name] = (
                            _choose_parent_id(parent_ids, stress_mode) if parent_ids else 1
                        )
                    else:
                        row[col.name] = random.randint(1, rows_per_table)
                elif (
                    i == 0
                    and tname in table_primary_alias
                    and col.name in z3_vars.get(table_primary_alias[tname], {})
                ):
                    # 第一行用 Z3 解
                    alias = table_primary_alias[tname]
                    z3_val = model.eval(z3_vars[alias][col.name])
                    row[col.name] = _z3_val_to_python(z3_val, col.col_type)
                else:
                    row[col.name] = _random_value(
                        col,
                        stress_mode=stress_mode,
                        pool=value_pools.get(col.name),
                    )
            rows.append(row)

        data[tname] = rows

    return data


def _condition_to_z3(cond: Condition, z3_vars: dict, alias_map: dict):
    """把 IR 条件节点转成 Z3 表达式。只处理数值比较，其余返回 None。"""
    try:
        import z3
    except ImportError:
        return None

    if isinstance(cond, Compare):
        # 解析左侧字段
        left_var = _resolve_z3_var(cond.field, z3_vars, alias_map)
        if left_var is None:
            return None

        # 右值：支持字面量数值，或另一侧列引用（常见于 Join on）
        if isinstance(cond.value, str) and "." in cond.value:
            right_val = _resolve_z3_var(cond.value, z3_vars, alias_map)
            if right_val is None:
                return None
        elif isinstance(cond.value, (int, float)):
            right_val = float(cond.value)
        else:
            return None

        op = cond.op
        if op == CmpOp.EQ:
            return left_var == right_val
        elif op == CmpOp.NEQ:
            return left_var != right_val
        elif op == CmpOp.GT:
            return left_var > right_val
        elif op == CmpOp.GTE:
            return left_var >= right_val
        elif op == CmpOp.LT:
            return left_var < right_val
        elif op == CmpOp.LTE:
            return left_var <= right_val

    elif isinstance(cond, And):
        l = _condition_to_z3(cond.left,  z3_vars, alias_map)
        r = _condition_to_z3(cond.right, z3_vars, alias_map)
        if l is not None and r is not None:
            import z3
            return z3.And(l, r)
        return l or r

    elif isinstance(cond, Or):
        l = _condition_to_z3(cond.left,  z3_vars, alias_map)
        r = _condition_to_z3(cond.right, z3_vars, alias_map)
        if l is not None and r is not None:
            import z3
            return z3.Or(l, r)
        return l or r

    elif isinstance(cond, Not):
        child = _condition_to_z3(cond.child, z3_vars, alias_map)
        if child is not None:
            import z3
            return z3.Not(child)

    return None


def _resolve_z3_var(field_name: str, z3_vars: dict, alias_map: dict):
    """
    把字段名解析成 Z3 变量。
    "o.amount" → z3_vars["o"]["amount"]
    "amount"   → 在所有表里搜索
    """
    if "." in field_name:
        alias, col_name = field_name.split(".", 1)
        if alias in alias_map and alias in z3_vars and col_name in z3_vars[alias]:
            return z3_vars[alias][col_name]
    else:
        matches = []
        for cols in z3_vars.values():
            if field_name in cols:
                matches.append(cols[field_name])
        if len(matches) == 1:
            return matches[0]
    return None


def _z3_val_to_python(z3_val, col_type: ColType) -> Any:
    """把 Z3 模型里的值转成 Python 值。"""
    try:
        import z3
        if z3.is_rational_value(z3_val):
            num = z3_val.numerator_as_long()
            den = z3_val.denominator_as_long()
            val = num / den
        elif z3.is_int_value(z3_val):
            val = z3_val.as_long()
        else:
            val = float(str(z3_val))

        if col_type == ColType.INT:
            return max(1, min(100, int(val)))
        else:
            return max(1.0, min(100.0, round(float(val), 1)))
    except Exception:
        return random.randint(1, 100)


# ---------------------------------------------------------------------------
# IR 遍历工具
# ---------------------------------------------------------------------------

def _extract_z3_constraints(ir: QueryNode) -> List[Condition]:
    """
    从 IR 树中提取适合交给 Z3 的简单约束。

    当前包含：
      - Filter 条件
      - Join on 的等值条件

    当前仍跳过：
      - Having 中引用聚合别名的复杂条件
    """
    result = []
    if isinstance(ir, Filter):
        result.append(ir.condition)
        result.extend(_extract_z3_constraints(ir.child))
    elif isinstance(ir, Having):
        # Having 条件引用聚合别名，Z3 处理较复杂，暂时跳过
        result.extend(_extract_z3_constraints(ir.child))
    elif isinstance(ir, (Project, GroupBy)):
        result.extend(_extract_z3_constraints(ir.child))
    elif isinstance(ir, Join):
        result.append(ir.on)
        result.extend(_extract_z3_constraints(ir.left))
        result.extend(_extract_z3_constraints(ir.right))
    elif isinstance(ir, Scan):
        pass
    return result


def _extract_alias_map(ir: QueryNode) -> Dict[str, str]:
    """从 IR 树中提取 alias -> table_name 的显式映射。"""
    result: Dict[str, str] = {}
    if isinstance(ir, Scan):
        result[ir.alias] = ir.table
    elif isinstance(ir, Join):
        result.update(_extract_alias_map(ir.left))
        result.update(_extract_alias_map(ir.right))
    elif hasattr(ir, "child"):
        result.update(_extract_alias_map(ir.child))
    return result


def _extract_tables(ir: QueryNode) -> List[str]:
    """从 IR 树中提取所有涉及的表名。"""
    result = []
    if isinstance(ir, Scan):
        result.append(ir.table)
    elif isinstance(ir, Join):
        result.extend(_extract_tables(ir.left))
        result.extend(_extract_tables(ir.right))
    elif hasattr(ir, "child"):
        result.extend(_extract_tables(ir.child))
    return list(dict.fromkeys(result))   # 去重保序


# ---------------------------------------------------------------------------
# 插入数据库
# ---------------------------------------------------------------------------

def _insert_all(schema: Schema, data: TableData) -> None:
    """把 TableData 里的所有数据插入数据库。"""
    from db.connector import insert_rows

    from generator.schema_gen import generate_drop_sqls
    ordered_names = list(reversed(generate_drop_sqls(schema)))

    for tname in ordered_names:
        if tname not in data or not data[tname]:
            continue
        table  = schema.get_table(tname)
        cols   = [c.name for c in table.columns]
        rows   = [tuple(row[c] for c in cols) for row in data[tname]]
        insert_rows(tname, cols, rows)


# ---------------------------------------------------------------------------
# 直接运行时：演示两种生成策略
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from generator.schema_gen import (
        generate_schema, generate_create_sqls,
        generate_drop_sqls, print_schema,
    )
    from generator.ir_gen import generate_ir
    from db.connector import (
        init_database, create_tables, drop_tables, execute_sql,
    )
    from ir.nodes import pretty_print

    print("=== 准备数据库 ===")
    init_database()

    # 生成 schema
    schema = generate_schema(num_tables=2, cols_per_table=3, fk_prob=1.0, seed=42)
    print_schema(schema)

    # 建表
    drop_tables(generate_drop_sqls(schema))
    create_tables(generate_create_sqls(schema))

    # 生成 IR
    ir, ctx = generate_ir(schema, seed=1)
    print("\n生成的 IR：")
    print(pretty_print(ir))

    # ------------------------------------------------------------------
    # 策略一：随机生成
    # ------------------------------------------------------------------
    print("\n=== 策略一：随机生成 ===")
    data = generate_random_only(schema, rows_per_table=5, seed=42)
    for tname, rows in data.items():
        print(f"\n表 {tname}（{len(rows)} 行）：")
        for r in rows:
            print(f"  {r}")

    # 验证数据已插入
    for t in schema.tables:
        rows = execute_sql(f"SELECT COUNT(*) AS cnt FROM `{t.name}`;")
        print(f"[验证] {t.name}: {rows[0]['cnt']} 行")

    # ------------------------------------------------------------------
    # 策略二：Z3 生成（如果安装了 z3-solver）
    # ------------------------------------------------------------------
    print("\n=== 策略二：Z3 生成 ===")
    # 清空数据
    for t in schema.tables:
        execute_sql(f"TRUNCATE TABLE `{t.name}`;")

    data2 = generate_and_insert(
        schema, ir,
        rows_per_table=5,
        use_z3=True,
        z3_timeout=5,
        seed=42,
    )
    for tname, rows in data2.items():
        print(f"\n表 {tname}（{len(rows)} 行）：")
        for r in rows:
            print(f"  {r}")

    # 清理
    print("\n=== 清理 ===")
    drop_tables(generate_drop_sqls(schema))
    print("完成 ✓")
