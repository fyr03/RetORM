"""
generator/schema_gen.py

随机生成数据库 Schema。

输出的 Schema 是一个纯 Python 数据结构，描述：
  - 有哪些表
  - 每张表有哪些列（列名、类型、是否可为 NULL）
  - 主键是什么
  - 外键关系（哪张表的哪列指向哪张表的主键）

同时提供：
  - 生成建表 SQL 的函数（给 connector 执行）
  - 生成 DROP TABLE SQL 的函数（按依赖顺序）
"""

import random
import string
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


# ---------------------------------------------------------------------------
# 数据类型定义
# ---------------------------------------------------------------------------

class ColType(Enum):
    INT    = "INT"
    FLOAT  = "FLOAT"
    VARCHAR = "VARCHAR(64)"
    # 后续可追加 DATE、BOOL 等


@dataclass
class Column:
    name:     str
    col_type: ColType
    nullable: bool = True    # 是否允许 NULL
    is_pk:    bool = False   # 是否是主键（主键不允许 NULL）


@dataclass
class ForeignKey:
    """从 src_table.src_col 指向 ref_table.ref_col（通常是 id）"""
    src_table: str
    src_col:   str
    ref_table: str
    ref_col:   str = "id"


@dataclass
class TableSchema:
    name:    str
    columns: List[Column]           # 包含 id 列在内的所有列
    fks:     List[ForeignKey] = field(default_factory=list)

    def col_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def non_pk_columns(self) -> List[Column]:
        return [c for c in self.columns if not c.is_pk]

    def get_column(self, name: str) -> Optional[Column]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


@dataclass
class Schema:
    tables: List[TableSchema]

    def table_names(self) -> List[str]:
        return [t.name for t in self.tables]

    def get_table(self, name: str) -> Optional[TableSchema]:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    def fk_pairs(self) -> List[ForeignKey]:
        """返回所有外键关系。"""
        result = []
        for t in self.tables:
            result.extend(t.fks)
        return result


# ---------------------------------------------------------------------------
# 随机生成
# ---------------------------------------------------------------------------

# 用固定词表避免随机字符串导致的可读性问题
_TABLE_NAMES  = ["orders", "users", "products", "reviews",
                 "categories", "items", "payments", "logs"]
_COL_PREFIXES = ["num", "val", "cnt", "score", "price",
                 "age", "rate", "amount", "total", "size"]


def generate_schema(
    num_tables:    int = 2,
    cols_per_table: int = 3,   # 除 id 外的列数（每张表）
    fk_prob:       float = 0.4, # 每张表（非第一张）有外键的概率
    nullable_prob: float = 0.3, # 每列可为 NULL 的概率
    seed:          Optional[int] = None,
) -> Schema:
    """
    随机生成一个 Schema。

    参数：
        num_tables:     表的数量，建议 2-4
        cols_per_table: 每张表除 id 外的列数
        fk_prob:        每张非第一张表生成一个外键的概率
        nullable_prob:  每列允许 NULL 的概率
        seed:           随机种子，传入则结果可复现

    返回：
        Schema 对象
    """
    if seed is not None:
        random.seed(seed)

    # 从词表里随机选表名，保证不重复
    available = _TABLE_NAMES.copy()
    random.shuffle(available)
    table_names = available[:num_tables]

    tables: List[TableSchema] = []

    for i, tname in enumerate(table_names):
        columns = _generate_columns(tname, cols_per_table, nullable_prob)
        fks: List[ForeignKey] = []

        # 第一张表没有外键；后续表以 fk_prob 概率引用前面某张表
        if i > 0 and random.random() < fk_prob:
            ref_table = random.choice(tables)   # 引用已生成的表
            fk_col_name = f"{ref_table.name}_id"
            # 外键列：INT NOT NULL
            fk_col = Column(
                name=fk_col_name,
                col_type=ColType.INT,
                nullable=False,
            )
            columns.append(fk_col)
            fks.append(ForeignKey(
                src_table=tname,
                src_col=fk_col_name,
                ref_table=ref_table.name,
                ref_col="id",
            ))

        tables.append(TableSchema(name=tname, columns=columns, fks=fks))

    return Schema(tables=tables)


def _generate_columns(
    table_name:   str,
    num_cols:     int,
    nullable_prob: float,
) -> List[Column]:
    """
    为一张表生成列列表，第一列固定是 INT PRIMARY KEY id。
    """
    columns: List[Column] = [
        Column(name="id", col_type=ColType.INT, nullable=False, is_pk=True)
    ]

    # 用前缀 + 序号生成列名，保证不重复
    prefixes = random.sample(_COL_PREFIXES, min(num_cols, len(_COL_PREFIXES)))
    if num_cols > len(prefixes):
        # 不够就加序号
        prefixes += [f"col{j}" for j in range(num_cols - len(prefixes))]

    for prefix in prefixes[:num_cols]:
        col_type = random.choice(list(ColType))
        nullable = random.random() < nullable_prob
        columns.append(Column(
            name=prefix,
            col_type=col_type,
            nullable=nullable,
        ))

    return columns


# ---------------------------------------------------------------------------
# 生成 SQL
# ---------------------------------------------------------------------------

def generate_create_sqls(schema: Schema) -> List[str]:
    """
    按拓扑顺序生成 CREATE TABLE SQL 列表。
    被引用的表（父表）先建，引用表（子表）后建，满足外键约束。

    返回：SQL 字符串列表，直接传给 connector.create_tables()
    """
    ordered = _topological_sort(schema)
    sqls = []
    for tname in ordered:
        table = schema.get_table(tname)
        sqls.append(_table_to_sql(table))
    return sqls


def generate_drop_sqls(schema: Schema) -> List[str]:
    """
    按逆拓扑顺序生成 DROP TABLE SQL 列表。
    子表先 drop，父表后 drop。

    返回：表名列表，直接传给 connector.drop_tables()
    """
    ordered = _topological_sort(schema)
    return list(reversed(ordered))


def _table_to_sql(table: TableSchema) -> str:
    """把一张表的 TableSchema 转成 CREATE TABLE SQL。"""
    col_defs = []
    for col in table.columns:
        null_str = "NOT NULL" if (not col.nullable or col.is_pk) else "NULL"
        col_def  = f"  `{col.name}` {col.col_type.value} {null_str}"
        col_defs.append(col_def)

    # PRIMARY KEY
    col_defs.append(f"  PRIMARY KEY (`id`)")

    # FOREIGN KEY（先不加 FOREIGN KEY 约束，避免测试时顺序问题）
    # 外键关系只在 IR 生成器里用于构造 JOIN 条件，不在 DDL 里强制

    body = ",\n".join(col_defs)
    return (
        f"CREATE TABLE IF NOT EXISTS `{table.name}` (\n"
        f"{body}\n"
        f");"
    )


def _topological_sort(schema: Schema) -> List[str]:
    """
    按外键依赖做拓扑排序，保证父表排在子表前面。
    如果没有外键，保持原始顺序。
    """
    # 建立依赖图：table → 它依赖的表的集合
    deps: Dict[str, set] = {t.name: set() for t in schema.tables}
    for fk in schema.fk_pairs():
        deps[fk.src_table].add(fk.ref_table)

    ordered = []
    visited = set()

    def visit(name: str):
        if name in visited:
            return
        visited.add(name)
        for dep in deps.get(name, []):
            visit(dep)
        ordered.append(name)

    for t in schema.tables:
        visit(t.name)

    return ordered


# ---------------------------------------------------------------------------
# 调试输出
# ---------------------------------------------------------------------------

def print_schema(schema: Schema) -> None:
    """打印 Schema 的人类可读描述。"""
    print(f"Schema（{len(schema.tables)} 张表）")
    for table in schema.tables:
        print(f"\n  表: {table.name}")
        for col in table.columns:
            pk_str   = " [PK]" if col.is_pk else ""
            null_str = " NULL" if col.nullable else " NOT NULL"
            print(f"    {col.name}: {col.col_type.value}{pk_str}{null_str}")
        for fk in table.fks:
            print(f"    FK: {fk.src_col} → {fk.ref_table}.{fk.ref_col}")


# ---------------------------------------------------------------------------
# 直接运行时：生成几个 Schema 并打印建表 SQL
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=== 示例一：固定 seed，2 张表 ===")
    s1 = generate_schema(num_tables=2, cols_per_table=3, seed=42)
    print_schema(s1)
    print("\n建表 SQL：")
    for sql in generate_create_sqls(s1):
        print(sql)
        print()

    print("\n=== 示例二：固定 seed，3 张表 ===")
    s2 = generate_schema(num_tables=3, cols_per_table=2, seed=7)
    print_schema(s2)
    print("\nDROP 顺序：", generate_drop_sqls(s2))

    print("\n=== 示例三：随机 seed，检查外键生成 ===")
    for i in range(3):
        s = generate_schema(num_tables=3, cols_per_table=2, fk_prob=0.8, seed=i*10)
        fks = s.fk_pairs()
        print(f"  seed={i*10}: 表={s.table_names()}, 外键数={len(fks)}", end="")
        if fks:
            fk = fks[0]
            print(f"  ({fk.src_table}.{fk.src_col} → {fk.ref_table}.{fk.ref_col})", end="")
        print()