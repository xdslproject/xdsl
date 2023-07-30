from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    TypeAlias,
    TypeVar,
)

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import (
    Attribute,
    IRNode,
    Operation,
    OperationInvT,
    OpResult,
    SSAValue,
)


class MatchContext:
    ctx: dict[str, Any]

    def __init__(self, ctx: dict[str, Any] | None = None) -> None:
        self.ctx = ctx or {}


Match: TypeAlias = dict[str, IRNode]

_T = TypeVar("_T")
_TCov = TypeVar("_TCov", covariant=True)


class Variable(Generic[_TCov], abc.ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def get(self, ctx: MatchContext) -> _TCov:
        return ctx.ctx[self.name]

    def set(self: Variable[_T], ctx: MatchContext, val: _T) -> bool:
        if self.name in ctx.ctx:
            return val == ctx.ctx[self.name]
        else:
            ctx.ctx[self.name] = val
            return True

    def __repr__(self) -> str:
        return f'{type(self).__name__}("{self.name}")'

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Variable) and self.name == __value.name

    def __hash__(self) -> int:
        return hash(self.name)


class OperationVariable(Variable[Operation]):
    ...


class AttributeVariable(Variable[Attribute]):
    ...


class SSAValueVariable(Variable[SSAValue]):
    ...


class OpResultVariable(Variable[OpResult]):
    ...


class Constraint(abc.ABC):
    @abc.abstractmethod
    def match(self, ctx: MatchContext) -> bool:
        raise NotImplementedError


@dataclass
class EqConstraint(Constraint):
    lhs_var: Variable[Any]
    rhs_var: Variable[Any]

    def match(self, ctx: MatchContext) -> bool:
        val = self.lhs_var.get(ctx)
        return self.rhs_var.set(ctx, val)


@dataclass
class TypeConstraint(Generic[_T], Constraint):
    var: Variable[_T]
    type: type[_T]

    def __init__(self, var: Variable[_T], type: type[_T]) -> None:
        self.var = var
        self.type = type

    def match(self, ctx: MatchContext) -> bool:
        op = self.var.get(ctx)
        return isinstance(op, self.type)


@dataclass
class AttributeValueConstraint(Constraint):
    attr_var: AttributeVariable
    attr: Attribute

    def match(self, ctx: MatchContext) -> bool:
        attr = self.attr_var.get(ctx)
        return attr == self.attr


@dataclass
class OperationAttributeConstraint(Constraint):
    op_var: OperationVariable
    attr_name: str
    attr_var: AttributeVariable

    def match(self, ctx: MatchContext) -> bool:
        a_obj = self.op_var.get(ctx)
        b_obj = getattr(a_obj, self.attr_name)
        return self.attr_var.set(ctx, b_obj)


@dataclass
class OperationOperandConstraint(Constraint):
    op_var: OperationVariable
    operand_name: str
    operand_var: Variable[SSAValue]

    def match(self, ctx: MatchContext) -> bool:
        a_obj = self.op_var.get(ctx)
        b_obj = getattr(a_obj, self.operand_name)
        return self.operand_var.set(ctx, b_obj)


@dataclass
class OperationResultConstraint(Constraint):
    op_var: OperationVariable
    res_name: str
    res_var: Variable[OpResult]

    def match(self, ctx: MatchContext) -> bool:
        a_obj = self.op_var.get(ctx)
        b_obj = getattr(a_obj, self.res_name)
        return self.res_var.set(ctx, b_obj)


@dataclass
class OpResultOpConstraint(Constraint):
    op_result_var: Variable[SSAValue]
    op_var: Variable[Operation]

    def match(self, ctx: MatchContext) -> bool:
        op_result = self.op_result_var.get(ctx)
        assert isinstance(op_result, OpResult)
        op = op_result.op
        return self.op_var.set(ctx, op)


class Query:
    match_variable_names: tuple[str, ...]
    variables: OrderedDict[str, Variable[Any]]
    constraints: list[Constraint]
    var_id: int = 0

    def __init__(
        self,
        match_variable_names: tuple[str, ...],
        variables: Iterable[Variable[Any]],
        constraints: Iterable[Constraint],
    ):
        self.match_variable_names = match_variable_names
        self.variables = OrderedDict()
        self.constraints = list(constraints)

        for var in variables:
            self.add_variable(var)

    def add_variable(self, var: Variable[Any]):
        assert var.name not in self.variables
        self.variables[var.name] = var

    @staticmethod
    def root(root_type: type[OperationInvT]) -> Query:
        root_var = OperationVariable("root")
        return Query(
            (root_var.name,), [root_var], [TypeConstraint(root_var, root_type)]
        )

    def match(self, operation: Operation) -> Match | None:
        """
        Returns a dictionary
        """
        ctx = MatchContext({"root": operation})

        for constraint in self.constraints:
            if not constraint.match(ctx):
                return None

        # all variables must have a value associated

        match = {name: ctx.ctx[name] for name in self.match_variable_names}

        assert set(self.match_variable_names) == set(match)

        return match

    def matches(self, module: ModuleOp) -> Iterator[Match]:
        for op in module.walk():
            if (env := self.match(op)) is not None:
                yield env

    def next_var_id(self) -> str:
        id = self.var_id
        self.var_id = id + 1
        return f"{id}"
