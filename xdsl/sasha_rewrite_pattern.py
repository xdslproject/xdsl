from __future__ import annotations

import abc
import inspect
from collections import OrderedDict
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
    TypeGuard,
    TypeVar,
    cast,
)

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.ir.core import Attribute, IRNode, Operation, OperationInvT, OpResult, SSAValue
from xdsl.irdl import IRDLOperation, OpDef
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

    def get(self, ctx: MatchContext) -> _T:
        return ctx.ctx[self.name]

    def set(self, ctx: MatchContext, val: _T) -> bool:
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
    op_result_var: Variable[OpResult]
    op_var: Variable[Operation]

    def match(self, ctx: MatchContext) -> bool:
        op_result = self.op_result_var.get(ctx)
        op = op_result.op
        return self.op_var.set(ctx, op)


class Query:
    variables: OrderedDict[str, Variable[Any]]
    constraints: list[Constraint]
    var_id: int = 0

    def __init__(
        self, variables: Iterable[Variable[Any]], constraints: Iterable[Constraint]
    ):
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
        return Query([root_var], [TypeConstraint(root_var, root_type)])

    def match(self, operation: Operation) -> Match | None:
        """
        Returns a dictionary
        """
        ctx = MatchContext({"root": operation})

        for constraint in self.constraints:
            if not constraint.match(ctx):
                return None

        # all variables must have a value associated

        match = {name: ctx.ctx[name] for name in self.variables}

        assert set(self.variables) == set(match)

        return match

    def matches(self, module: ModuleOp) -> Iterator[Match]:
        for op in module.walk():
            if (env := self.match(op)) is not None:
                yield env

    def next_var_id(self) -> str:
        id = self.var_id
        self.var_id = id + 1
        return f"{id}"


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


@dataclass
class _QBVC:
    """
    Query builder variable contents
    """

    var: Variable[Any]
    query: Query
    property_variables: dict[str, _QueryBuilderVariable[_QBVC]]

    def register_var(self) -> bool:
        """
        Returns False if the variable was already registered.
        """
        if self.var.name in self.query.variables:
            return False
        self.query.add_variable(self.var)
        for var in self.property_variables.values():
            var.qbvc__.register_var()
        return True

    def constrain_type(self, hint: type[_T]) -> TypeGuard[_T]:
        if self.var.name not in self.query.variables:
            self.query.add_variable(self.var)
        self.query.constraints.append(TypeConstraint(self.var, hint))
        return True

    def eq(self, self_variable: _QueryBuilderVariable[_QBVC], value: Any) -> bool:
        if isinstance(value, _QueryBuilderVariable):
            other_variable = cast(_QueryBuilderVariable[_QBVC], value)
            return self.eq_variable(self_variable, other_variable)
        else:
            return self.eq_value(self_variable, value)

    def eq_variable(
        self,
        self_variable: _QueryBuilderVariable[_QBVC],
        other_variable: _QueryBuilderVariable[_QBVC],
    ) -> bool:
        # Constrain the two variables to be equal
        assert self.var.name in self.query.variables
        other_qbvc = other_variable.qbvc__
        self.query.constraints.append(EqConstraint(self.var, other_qbvc.var))
        other_qbvc.register_var()
        return True

    def eq_value(self, self_variable: _QueryBuilderVariable[_QBVC], value: Any) -> bool:
        return False

    def get_attribute(self, name: str) -> _QueryBuilderVariable[_QBVC] | None:
        return None


_QBVCT = TypeVar("_QBVCT", bound=_QBVC)
_QBVCTCov = TypeVar("_QBVCTCov", bound=_QBVC, covariant=True)


class _QueryBuilderVariable(Generic[_QBVCTCov]):
    qbvc__: _QBVCTCov
    """
    Very unlikely attribute name for a class we might encounter, holds state of the
    variable.
    """

    def __init__(self, qbvc: _QBVCTCov) -> None:
        self.qbvc__ = qbvc

    def __getattribute__(self, __name: str) -> Any:
        qbvc = cast(_QBVC, super().__getattribute__("qbvc__"))
        if __name == "qbvc__":
            return qbvc
        else:
            attr = qbvc.property_variables.get(__name)
            if attr is None:
                attr = qbvc.get_attribute(__name)
                if attr is None:
                    raise AttributeError
                # register property in this variable's cache
                qbvc.property_variables[__name] = attr
                # register property's var in query if this one is already registered
                if qbvc.var.name in qbvc.query.variables:
                    qbvc.query.add_variable(attr.qbvc__.var)

            return attr

    def __eq__(self, __value: object) -> bool:
        return self.qbvc__.eq(self, __value)


@dataclass
class _OperationQBVC(_QBVC):
    def get_attribute(self, name: str) -> Any:
        # TODO: add operation properties here
        return None


@dataclass
class _IRDLOperationQBVC(_OperationQBVC):
    cls: type[IRDLOperation]
    var: OperationVariable

    @property
    def op_def(self) -> OpDef:
        return self.cls.irdl_definition

    def register_var(self) -> bool:
        did_register = super().register_var()
        if did_register:
            self.query.constraints.append(TypeConstraint(self.var, self.cls))
        return did_register

    def get_attribute(self, name: str) -> _QueryBuilderVariable[_QBVC] | None:
        if name in dict(self.op_def.operands):
            new_var = SSAValueVariable(self.query.next_var_id())
            new_qbvc = _SSAValueQBVC(new_var, self.query, {})
            self.query.constraints.append(
                OperationOperandConstraint(self.var, name, new_var)
            )
            return _QueryBuilderVariable(new_qbvc)
        elif name in dict(self.op_def.results):
            assert False
        else:
            assert False


class _SSAValueQBVC(_QBVC):
    var: OpResultVariable

    def get_attribute(self, name: str) -> _QueryBuilderVariable[_QBVC] | None:
        if name == "op":
            new_qbvc = _OperationQBVC(
                OperationVariable(self.query.next_var_id()), self.query, {}
            )
            new_var = _QueryBuilderVariable(new_qbvc)
            self.query.constraints.append(OpResultOpConstraint(self.var, new_qbvc.var))
            return new_var


_P = ParamSpec("_P")


class PatternQuery(Generic[_P], Query):
    pass


def rewrite_pattern_query(func: Callable[_P, None]) -> PatternQuery[_P]:
    params = list(inspect.signature(func).parameters.items())

    assert "root" in (name for name, _ in params)

    query = PatternQuery((), ())
    fake_vars: dict[str, _QueryBuilderVariable[_QBVC]] = {}

    for name, param in params:
        cls = param.annotation
        if issubclass(cls, IRDLOperation):
            # Don't add the variables here, they will be added as they are traversed
            var = OperationVariable(name)
            qbvc = _IRDLOperationQBVC(var, query, {}, cls)
            fake_vars[name] = _QueryBuilderVariable(qbvc)

    # The root is given every time, so we add it type check immediately
    fake_vars["root"].qbvc__.register_var()

    func(**fake_vars)  # pyright: ignore[reportGeneralTypeIssues]

    return query


def query_rewrite_pattern(
    query: PatternQuery[_P],
) -> Callable[[Callable[Concatenate[PatternRewriter, _P], None]], QueryRewritePattern]:
    def impl(
        func: Callable[Concatenate[PatternRewriter, _P], None]
    ) -> QueryRewritePattern:
        def rewrite(match: Match, rewriter: PatternRewriter) -> None:
            return func(rewriter, **match)  # pyright: ignore[reportGeneralTypeIssues]

        return QueryRewritePattern(query, rewrite)

    return impl
