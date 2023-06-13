from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    Iterator,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

# cast,
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.ir.core import Attribute, IRNode, Operation, OperationInvT
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern


class MatchContext:
    ctx: dict[str, Any]

    def __init__(self, ctx: dict[str, Any] | None = None) -> None:
        self.ctx = ctx or {}


Match: TypeAlias = dict[str, IRNode]

_T = TypeVar("_T")


class Variable(Generic[_T], abc.ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def get(self, ctx: MatchContext) -> _T | None:
        return ctx.ctx[self.name]

    def set(self, ctx: MatchContext, val: _T) -> bool:
        if self.name in ctx.ctx:
            return val == ctx.ctx[self.name]
        else:
            ctx.ctx[self.name] = val
            return True


class OperationVariable(Variable[Operation]):
    ...


class AttributeVariable(Variable[Attribute]):
    ...


class AnyVariable(Variable[Any]):
    ...


class Constraint(abc.ABC):
    @abc.abstractmethod
    def match(self, ctx: MatchContext) -> bool:
        raise NotImplementedError


class OpTypeConstraint(Constraint):
    op_var: OperationVariable
    op_type: type[Operation]

    def __init__(self, op_var: OperationVariable, op_type: type[Operation]) -> None:
        self.op_var = op_var
        self.op_type = op_type

    def match(self, ctx: MatchContext) -> bool:
        op = self.op_var.get(ctx)
        return isinstance(op, self.op_type)


@dataclass
class AttributeValueConstraint(Constraint):
    attr_var: AttributeVariable
    attr: Attribute

    def match(self, ctx: MatchContext) -> bool:
        attr = self.attr_var.get(ctx)
        return attr == self.attr


@dataclass
class PropertyConstraint(Constraint):
    a: Variable[Any]
    property_name: str
    b: Variable[Any]

    def match(self, ctx: MatchContext) -> bool:
        a_obj = self.a.get(ctx)
        b_obj = getattr(a_obj, self.property_name)
        return self.b.set(ctx, b_obj)


class Query:
    variables: list[Variable[Any]]
    constraints: list[Constraint]
    var_id: int = 0

    def __init__(
        self, variables: Iterable[Variable[Any]], constraints: Iterable[Constraint]
    ):
        self.variables = list(variables)
        self.constraints = list(constraints)

    @staticmethod
    def root(root_type: type[OperationInvT]) -> Query:
        root_var = OperationVariable("root")
        return Query([root_var], [OpTypeConstraint(root_var, root_type)])

    def match(self, operation: Operation) -> Match | None:
        """
        Returns a dictionary
        """
        ctx = MatchContext({"root": operation})

        for constraint in self.constraints:
            if not constraint.match(ctx):
                return None

        # all variables must have a value associated
        var_names = set(var.name for var in self.variables)

        match = {name: ctx.ctx[name] for name in var_names}

        assert var_names == set(match)

        return match

    def matches(self, module: ModuleOp) -> Iterator[Match]:
        for op in module.walk():
            if (env := self.match(op)) is not None:
                yield env

    def next_var_id(self) -> int:
        id = self.var_id
        self.var_id = id + 1
        return id


class QueryRewritePattern(RewritePattern):
    query: Query
    rewrite: Callable[[Match, PatternRewriter], None]

    def __init__(
        self, query: Query, rewrite: Callable[[Match, PatternRewriter], None]
    ) -> None:
        self.query = query
        self.rewrite = rewrite

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if (match := self.query.match(op)) is not None:
            self.rewrite(match, rewriter)


_P = ParamSpec("_P")


def query_rewrite_pattern(
    func: Callable[Concatenate[bool, PatternRewriter, OperationInvT, _P], None]
) -> QueryRewritePattern:
    params = list(inspect.signature(func).parameters.items())

    assert params[0][0] == "match"
    assert params[0][1] == bool

    assert params[1][0] == "rewriter"
    assert params[1][1] == PatternRewriter

    assert params[2][0] == "root"

    query = Query.root(params[2][1].annotation)

    for name, param in params[3:]:
        assert False
        cls = param.annotation
        if issubclass(cls, Operation):
            query.variables.append(OperationVariable(name))

    # fake_rewriter = cast(PatternRewriter, None)

    def rewrite(match: Match, rewriter: PatternRewriter) -> None:
        root = match["root"]
        return func(
            False, rewriter, root, **match  # pyright: ignore[reportGeneralTypeIssues]
        )

    return QueryRewritePattern(query, rewrite)
