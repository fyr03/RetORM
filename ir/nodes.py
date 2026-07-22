"""
ir/nodes.py

RetORM 的查询语义中间表示（IR）。
所有节点都是纯数据结构，不包含任何执行逻辑。
三条翻译路径（python_ref / sql / sqlalchemy_orm）都以这里的节点为输入。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum


# ---------------------------------------------------------------------------
# 基础枚举
# ---------------------------------------------------------------------------

class AggFunc(Enum):
    """聚合函数类型"""
    SUM   = "SUM"
    COUNT = "COUNT"
    AVG   = "AVG"
    MAX   = "MAX"
    MIN   = "MIN"


class CmpOp(Enum):
    """比较运算符"""
    EQ  = "="
    NEQ = "!="
    GT  = ">"
    GTE = ">="
    LT  = "<"
    LTE = "<="


class JoinType(Enum):
    """JOIN 类型，目前只用 INNER，后续可扩展 LEFT"""
    INNER = "INNER"
    LEFT  = "LEFT"


# ---------------------------------------------------------------------------
# 条件表达式节点
# ---------------------------------------------------------------------------

@dataclass
class Compare:
    """
    单个比较条件，对应 SQL 里的 field op value。
    field: 列名，格式为 "table.column" 或 "column"
    op:    比较运算符
    value: 右值，可以是 Python 字面量（int / float / str / None）
    
    例：Compare("orders.amount", CmpOp.GT, 100)
        → orders.amount > 100
    """
    field: str
    op: CmpOp
    value: Union[int, float, str, None]


@dataclass
class And:
    """
    逻辑与，对应 SQL 里的 p1 AND p2。
    
    例：And(Compare("age", CmpOp.GT, 18), Compare("age", CmpOp.LT, 60))
        → age > 18 AND age < 60
    """
    left:  "Condition"
    right: "Condition"


@dataclass
class Or:
    """
    逻辑或，对应 SQL 里的 p1 OR p2。
    """
    left:  "Condition"
    right: "Condition"


@dataclass
class Not:
    """
    逻辑非，对应 SQL 里的 NOT p。
    """
    child: "Condition"


# 条件节点的类型别名，方便类型标注
Condition = Union[Compare, And, Or, Not]


# ---------------------------------------------------------------------------
# 聚合表达式节点
# ---------------------------------------------------------------------------

@dataclass
class Aggregate:
    """
    聚合表达式，只能出现在 GroupBy 之后的上下文中。
    
    func:  聚合函数类型
    field: 被聚合的列名，COUNT(*) 时可以传 "*"
    alias: 结果列的别名，用于 Project 和 Having 引用
    
    例：Aggregate(AggFunc.SUM, "orders.amount", "total_amount")
        → SUM(orders.amount) AS total_amount
    """
    func:  AggFunc
    field: str
    alias: str


# ---------------------------------------------------------------------------
# 查询节点（IR 树的主体）
# ---------------------------------------------------------------------------

@dataclass
class Scan:
    """
    扫描一张表，是所有 IR 树的叶节点（起点）。
    
    table: 表名
    alias: 表别名，用于 Join 时区分两张表的列
    
    例：Scan("orders", "o")
        → FROM orders AS o（SQL 路径）
          或直接读 orders 表的所有行（程序逻辑路径）
    """
    table: str
    alias: Optional[str] = None

    def __post_init__(self):
        # 如果没有显式指定别名，默认用表名本身
        if self.alias is None:
            self.alias = self.table


@dataclass
class Filter:
    """
    按条件过滤，对应 SQL 里的 WHERE 子句。
    必须放在 GroupBy 之前；GroupBy 之后的过滤用 Having。
    
    condition: 过滤条件，可以是任意 Condition 节点
    child:     子查询节点
    
    例：Filter(Compare("amount", CmpOp.GT, 0), Scan("orders"))
        → SELECT * FROM orders WHERE amount > 0
    """
    condition: Condition
    child: "QueryNode"


@dataclass
class Join:
    """
    内连接两张表，对应 SQL 里的 INNER JOIN。
    目前只支持两张表的等值连接，on 是一个等值条件。
    
    left:      左侧查询节点（通常是 Scan）
    right:     右侧查询节点（通常是 Scan）
    on:        连接条件，通常是 Compare(left_fk, CmpOp.EQ, right_pk)
               字段格式必须带表名前缀，例如 "orders.user_id"
    join_type: JOIN 类型，目前固定为 INNER
    
    例：Join(
            Scan("orders", "o"),
            Scan("users", "u"),
            Compare("o.user_id", CmpOp.EQ, "u.id")
        )
        → FROM orders AS o INNER JOIN users AS u ON o.user_id = u.id
    
    注意：on 条件的右值如果是另一张表的列名（字符串），
          翻译器需要识别出这是列引用而非字符串字面量。
          约定：右值为字符串且包含 "." 时视为列引用。
    """
    left:      "QueryNode"
    right:     "QueryNode"
    on:        Compare
    join_type: JoinType = JoinType.INNER


@dataclass
class GroupBy:
    """
    分组，对应 SQL 里的 GROUP BY 子句。
    Having 和 Aggregate 只能出现在 GroupBy 之后。
    
    fields:     分组字段列表，格式为 "table.column" 或 "column"
    aggregates: 该查询需要计算的聚合表达式列表
    child:      子查询节点
    
    例：GroupBy(
            fields=["user_id"],
            aggregates=[Aggregate(AggFunc.SUM, "amount", "total")],
            child=Scan("orders")
        )
        → SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id
    """
    fields:     List[str]
    aggregates: List[Aggregate]
    child:      "QueryNode"


@dataclass
class Having:
    """
    分组后过滤，对应 SQL 里的 HAVING 子句。
    必须放在 GroupBy 之后；Having 的条件里引用聚合结果时用 Aggregate 的 alias。
    
    condition: 过滤条件，可以引用聚合别名（如 "total"）或分组字段
    child:     子查询节点，必须是 GroupBy 节点
    
    例：Having(
            Compare("total", CmpOp.GT, 100),
            GroupBy(["user_id"], [Aggregate(AggFunc.SUM, "amount", "total")], Scan("orders"))
        )
        → ... GROUP BY user_id HAVING SUM(amount) > 100
    """
    condition: Condition
    child:     "QueryNode"  # 语义上要求 child 必须是 GroupBy


@dataclass
class Project:
    """
    投影，对应 SQL 里的 SELECT 指定列。
    放在 IR 树的最外层，决定最终返回哪些列。
    如果没有 Project 节点，默认返回所有列（SELECT *）。
    
    fields: 要返回的列名列表，可以是普通列名或聚合别名
    child:  子查询节点
    
    例：Project(["user_id", "total"], Having(...))
        → SELECT user_id, total FROM ...
    """
    fields: List[str]
    child:  "QueryNode"


# 查询节点的类型别名
QueryNode = Union[Scan, Filter, Join, GroupBy, Having, Project]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def pretty_print(node, indent: int = 0) -> str:
    """
    递归打印 IR 树，方便调试时肉眼检查结构。
    
    用法：print(pretty_print(my_ir))
    """
    pad = "  " * indent

    if isinstance(node, Scan):
        return f"{pad}Scan(table={node.table!r}, alias={node.alias!r})"

    elif isinstance(node, Filter):
        cond = _fmt_condition(node.condition)
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Filter(\n{pad}  condition={cond},\n{pad}  child=\n{child_str}\n{pad})"

    elif isinstance(node, Join):
        left_str  = pretty_print(node.left,  indent + 1)
        right_str = pretty_print(node.right, indent + 1)
        on_str    = _fmt_condition(node.on)
        return (f"{pad}Join(\n{pad}  type={node.join_type.value},\n{pad}  on={on_str},\n"
                f"{pad}  left=\n{left_str},\n"
                f"{pad}  right=\n{right_str}\n{pad})")

    elif isinstance(node, GroupBy):
        aggs = [f"Aggregate({a.func.value}, {a.field!r}, alias={a.alias!r})"
                for a in node.aggregates]
        child_str = pretty_print(node.child, indent + 1)
        return (f"{pad}GroupBy(\n{pad}  fields={node.fields},\n"
                f"{pad}  aggregates={aggs},\n"
                f"{pad}  child=\n{child_str}\n{pad})")

    elif isinstance(node, Having):
        cond      = _fmt_condition(node.condition)
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Having(\n{pad}  condition={cond},\n{pad}  child=\n{child_str}\n{pad})"

    elif isinstance(node, Project):
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Project(\n{pad}  fields={node.fields},\n{pad}  child=\n{child_str}\n{pad})"

    else:
        return f"{pad}{repr(node)}"


def _fmt_condition(cond) -> str:
    """内部辅助：把条件节点格式化成单行字符串。"""
    if isinstance(cond, Compare):
        return f"Compare({cond.field!r} {cond.op.value} {cond.value!r})"
    elif isinstance(cond, And):
        return f"And({_fmt_condition(cond.left)}, {_fmt_condition(cond.right)})"
    elif isinstance(cond, Or):
        return f"Or({_fmt_condition(cond.left)}, {_fmt_condition(cond.right)})"
    elif isinstance(cond, Not):
        return f"Not({_fmt_condition(cond.child)})"
    else:
        return repr(cond)


# ---------------------------------------------------------------------------
# 快速自检：直接运行这个文件时打印两个示例 IR
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 示例一：单表过滤 + 投影
    # SELECT user_id FROM orders WHERE amount > 100
    ir1 = Project(
        fields=["user_id"],
        child=Filter(
            condition=Compare("amount", CmpOp.GT, 100),
            child=Scan("orders")
        )
    )
    print("=== IR 示例一：单表过滤 + 投影 ===")
    print(pretty_print(ir1))
    print()

    # 示例二：GroupBy + Having + 投影
    # SELECT user_id, SUM(amount) AS total
    # FROM orders
    # GROUP BY user_id
    # HAVING SUM(amount) > 100
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
    print("=== IR 示例二：GroupBy + Having + 投影 ===")
    print(pretty_print(ir2))
    print()

    # 示例三：两表 Join + Filter
    # SELECT o.user_id, o.amount
    # FROM orders AS o
    # INNER JOIN users AS u ON o.user_id = u.id
    # WHERE u.age > 18
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
    print("=== IR 示例三：两表 Join + Filter ===")
    print(pretty_print(ir3))
