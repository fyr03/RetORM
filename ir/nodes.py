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


class ArithOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


@dataclass
class ArithExpr:
    left: "ValueExpr"
    op: ArithOp
    right: "ValueExpr"


@dataclass
class Compare:
    field: "ValueExpr"
    op: CmpOp
    value: "ValueExpr"


@dataclass
class InList:
    field: "ValueExpr"
    values: List["ValueExpr"]
    negated: bool = False


@dataclass
class Between:
    field: "ValueExpr"
    lower: "ValueExpr"
    upper: "ValueExpr"
    negated: bool = False


@dataclass
class Like:
    field: "ValueExpr"
    pattern: str
    negated: bool = False


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


@dataclass
class WhenClause:
    condition: "Condition"
    value: "ValueExpr"


@dataclass
class CaseWhen:
    cases: List[WhenClause]
    else_value: "ValueExpr"


ValueExpr = Union[str, int, float, None, ArithExpr, CaseWhen]
Condition = Union[Compare, InList, Between, Like, And, Or, Not]


@dataclass
class Aggregate:
    func: AggFunc
    field: Union[str, ValueExpr]
    alias: str


@dataclass
class SelectItem:
    expr: ValueExpr
    alias: str


@dataclass
class OrderKey:
    field: Union[str, ValueExpr]
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
    fields: List[Union[str, ValueExpr]]
    aggregates: List[Aggregate]
    child: "QueryNode"


@dataclass
class Having:
    condition: Condition
    child: "QueryNode"


@dataclass
class Project:
    fields: List[Union[str, SelectItem]]
    child: "QueryNode"


@dataclass
class Distinct:
    child: "QueryNode"


@dataclass
class OrderBy:
    keys: List[OrderKey]
    child: "QueryNode"


@dataclass
class LimitOffset:
    limit: int
    offset: int = 0
    child: "QueryNode" = None


QueryNode = Union[Scan, Filter, Join, GroupBy, Having, Project, Distinct, OrderBy, LimitOffset]


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
            f"Aggregate({agg.func.value}, {_fmt_expr(agg.field)}, alias={agg.alias!r})"
            for agg in node.aggregates
        ]
        fields = [_fmt_expr(field) for field in node.fields]
        child_str = pretty_print(node.child, indent + 1)
        return (
            f"{pad}GroupBy(\n{pad}  fields={fields},\n"
            f"{pad}  aggregates={aggs},\n"
            f"{pad}  child=\n{child_str}\n{pad})"
        )

    if isinstance(node, Having):
        cond = _fmt_condition(node.condition)
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Having(\n{pad}  condition={cond},\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, Project):
        fields = [_fmt_project_field(field) for field in node.fields]
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Project(\n{pad}  fields={fields},\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, Distinct):
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}Distinct(\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, OrderBy):
        keys = [
            f"OrderKey({_fmt_expr(key.field)}, descending={key.descending})"
            for key in node.keys
        ]
        child_str = pretty_print(node.child, indent + 1)
        return f"{pad}OrderBy(\n{pad}  keys={keys},\n{pad}  child=\n{child_str}\n{pad})"

    if isinstance(node, LimitOffset):
        child_str = pretty_print(node.child, indent + 1)
        return (
            f"{pad}LimitOffset(\n{pad}  limit={node.limit},\n{pad}  offset={node.offset},\n"
            f"{pad}  child=\n{child_str}\n{pad})"
        )

    return f"{pad}{repr(node)}"


def _fmt_project_field(field) -> str:
    if isinstance(field, SelectItem):
        return f"SelectItem(expr={_fmt_expr(field.expr)}, alias={field.alias!r})"
    return _fmt_expr(field)


def _fmt_expr(expr) -> str:
    if isinstance(expr, ArithExpr):
        return f"ArithExpr({_fmt_expr(expr.left)} {expr.op.value} {_fmt_expr(expr.right)})"
    if isinstance(expr, CaseWhen):
        cases = [
            f"WhenClause(condition={_fmt_condition(case.condition)}, value={_fmt_expr(case.value)})"
            for case in expr.cases
        ]
        return f"CaseWhen(cases={cases}, else_value={_fmt_expr(expr.else_value)})"
    return repr(expr)


def _fmt_condition(cond) -> str:
    if isinstance(cond, Compare):
        return f"Compare({_fmt_expr(cond.field)} {cond.op.value} {_fmt_expr(cond.value)})"
    if isinstance(cond, InList):
        values = [_fmt_expr(value) for value in cond.values]
        prefix = "NOT " if cond.negated else ""
        return f"{prefix}InList(field={_fmt_expr(cond.field)}, values={values})"
    if isinstance(cond, Between):
        prefix = "NOT " if cond.negated else ""
        return (
            f"{prefix}Between(field={_fmt_expr(cond.field)}, "
            f"lower={_fmt_expr(cond.lower)}, upper={_fmt_expr(cond.upper)})"
        )
    if isinstance(cond, Like):
        prefix = "NOT " if cond.negated else ""
        return f"{prefix}Like(field={_fmt_expr(cond.field)}, pattern={cond.pattern!r})"
    if isinstance(cond, And):
        return f"And({_fmt_condition(cond.left)}, {_fmt_condition(cond.right)})"
    if isinstance(cond, Or):
        return f"Or({_fmt_condition(cond.left)}, {_fmt_condition(cond.right)})"
    if isinstance(cond, Not):
        return f"Not({_fmt_condition(cond.child)})"
    return repr(cond)
