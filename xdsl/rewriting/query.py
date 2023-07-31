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
_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")

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
class UnaryConstraint(Generic[_T], Constraint, abc.ABC):
    var: Variable[_T]


@dataclass
class BinaryConstraint(Generic[_T0, _T1], Constraint):
    var0: Variable[_T0]
    var1: Variable[_T1]


@dataclass
class EqConstraint(BinaryConstraint[Any, Any]):
    def match(self, ctx: MatchContext) -> bool:
        val = self.var0.get(ctx)
        return self.var1.set(ctx, val)


@dataclass
class TypeConstraint(UnaryConstraint[_T], Constraint):
    type: type[_T]

    def __init__(self, var: Variable[_T], type: type[_T]) -> None:
        self.var = var
        self.type = type

    def match(self, ctx: MatchContext) -> bool:
        op = self.var.get(ctx)
        return isinstance(op, self.type)


@dataclass
class AttributeValueConstraint(UnaryConstraint[Attribute]):
    attr: Attribute

    def match(self, ctx: MatchContext) -> bool:
        attr = self.var.get(ctx)
        return attr == self.attr


@dataclass
class PropertyConstraint(BinaryConstraint[_T0, _T1]):
    property_name: str

    def match(self, ctx: MatchContext) -> bool:
        val0 = self.var0.get(ctx)
        val1 = getattr(val0, self.property_name)
        return self.var1.set(ctx, val1)


SSAValueInvT = TypeVar("SSAValueInvT", bound=SSAValue)


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
