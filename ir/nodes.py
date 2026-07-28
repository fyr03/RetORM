"""
ir/nodes.py

RetORM query intermediate representation.

All nodes are pure data structures. The three execution paths
(`python_ref`, `sql`, `sqlalchemy_orm`) all consume the same IR.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class AggFunc(Enum):
    SUM = "SUM"
    COUNT = "COUNT"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"


class CmpOp(Enum):
    EQ = "="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="


class JoinType(Enum):
    INNER = "INNER"
    LEFT = "LEFT"


@dataclass
class Compare:
    field: str
    op: CmpOp
    value: Union[int, float, str, None]


@dataclass
class And:
    left: "Condition"
    right: "Condition"


@dataclass
class Or:
    left: "Condition"
    right: "Condition"


@dataclass
class Not:
    child: "Condition"


Condition = Union[Compare, And, Or, Not]


@dataclass
class Aggregate:
    func: AggFunc
    field: str
    alias: str


@dataclass
class OrderKey:
    field: str
    descending: bool = False


@dataclass
class Scan:
    table: str
    alias: Optional[str] = None

    def __post_init__(self):
        if self.alias is None:
            self.alias = self.table


@dataclass
class Filter:
    condition: Condition
    child: "QueryNode"


@dataclass
class Join:
    left: "QueryNode"
    right: "QueryNode"
    on: Compare
    join_type: JoinType = JoinType.INNER


@dataclass
class GroupBy:
    fields: List[str]
    aggregates: List[Aggregate]
    child: "QueryNode"


@dataclass
class Having:
    condition: Condition
    child: "QueryNode"


@dataclass
class Project:
    fields: List[str]
    child: "QueryNode"


@dataclass
class Distinct:
    child: "QueryNode"


@dataclass
class OrderBy:
    keys: List[OrderKey]
    child: "QueryNode"


QueryNode = Union[Scan, Filter, Join, GroupBy, Having, Project, Distinct, OrderBy]


def pretty_print(node, indent: int = 0) -> str:
    pad = "  " * indent

    if isinstance(node, Scan):
        return f"{pad}Scan(table={node.table!r}, alias={node.alias!r})"

    if isinstance(node, Filter):
        cond = _fmt_condition(node.condition)
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Filter(\n{pad}  condition={cond},\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, Join):
        left_str = pretty_print(node.left, indent + 1)
        right_str = pretty_print(node.right, indent + 1)
        on_str = _fmt_condition(node.on)
        return (
            f"{pad}Join(\n{pad}  type={node.join_type.value},\n{pad}  on={on_str},\n"
            f"{pad}  left=\n{left_str},\n"
            f"{pad}  right=\n{right_str}\n{pad})"
        )

    if isinstance(node, GroupBy):
        aggs = [
            f"Aggregate({agg.func.value}, {agg.field!r}, alias={agg.alias!r})"
            for agg in node.aggregates
        ]
        child_str = pretty_print(node.child, indent + 1)
        return (
            f"{pad}GroupBy(\n{pad}  fields={node.fields},\n"
            f"{pad}  aggregates={aggs},\n"
            f"{pad}  child=\n{child_str}\n{pad})"
        )

    if isinstance(node, Having):
        cond = _fmt_condition(node.condition)
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Having(\n{pad}  condition={cond},\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, Project):
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Project(\n{pad}  fields={node.fields},\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, Distinct):
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Distinct(\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, OrderBy):
        keys = [
            f"OrderKey({key.field!r}, descending={key.descending})"
            for key in node.keys
        ]
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}OrderBy(\n{pad}  keys={keys},\n{pad}  child=\n{child_str}\n{pad})"

    return f"{pad}{repr(node)}"


def _fmt_condition(cond) -> str:
    if isinstance(cond, Compare):
        return f"Compare({cond.field!r} {cond.op.value} {cond.value!r})"
    if isinstance(cond, And):
        return f"And({_fmt_condition(cond.left)}, {_fmt_condition(cond.right)})"
    if isinstance(cond, Or):
        return f"Or({_fmt_condition(cond.left)}, {_fmt_condition(cond.right)})"
    if isinstance(cond, Not):
        return f"Not({_fmt_condition(cond.child)})"
    return repr(cond)
