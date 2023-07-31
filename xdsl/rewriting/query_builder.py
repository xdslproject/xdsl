from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    TypeGuard,
    TypeVar,
    cast,
)

from xdsl.ir.core import Attribute
from xdsl.irdl import IRDLOperation, OpDef
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriting.query import (
    AttributeValueConstraint,
    AttributeVariable,
    EqConstraint,
    Match,
    OperationAttributeConstraint,
    OperationOperandConstraint,
    OperationVariable,
    OpResultOpConstraint,
    Query,
    SSAValueVariable,
    TypeConstraint,
    Variable,
)
from xdsl.rewriting.query_rewrite_pattern import QueryRewritePattern

_T = TypeVar("_T")


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
                OperationOperandConstraint(self.var, new_var, name)
            )
            return _QueryBuilderVariable(new_qbvc)
        elif name in dict(self.op_def.results):
            assert False, f"{name}"
        elif name in dict(self.op_def.attributes):
            new_var = AttributeVariable(self.query.next_var_id())
            new_qbvc = _AttributeQBVC(new_var, self.query, {})
            self.query.constraints.append(
                OperationAttributeConstraint(self.var, new_var, name)
            )
            return _QueryBuilderVariable(new_qbvc)
        else:
            assert False


class _SSAValueQBVC(_QBVC):
    var: SSAValueVariable

    def get_attribute(self, name: str) -> _QueryBuilderVariable[_QBVC] | None:
        if name == "op":
            new_qbvc = _OperationQBVC(
                OperationVariable(self.query.next_var_id()), self.query, {}
            )
            new_var = _QueryBuilderVariable(new_qbvc)
            self.query.constraints.append(OpResultOpConstraint(self.var, new_qbvc.var))
            return new_var


class _AttributeQBVC(_QBVC):
    var: AttributeVariable

    def eq(self, self_variable: _QueryBuilderVariable[_QBVC], value: Any) -> bool:
        assert isinstance(value, Attribute)
        self.query.constraints.append(AttributeValueConstraint(self.var, value))
        return True


QueryParams = ParamSpec("QueryParams")


class PatternQuery(Generic[QueryParams], Query):
    """
    Can only be created as an annottation.

    @PatternQuery
    def my_query(root: arith.Addi, rhs_input: arith.Constant):
        ...
    """

    def __init__(self, func: Callable[QueryParams, bool]):
        params = [
            (name, param.annotation)
            for name, param in inspect.signature(func).parameters.items()
        ]

        names = tuple(name for name, _ in params)

        assert "root" in names

        super().__init__(names, (), ())
        fake_vars: dict[str, _QueryBuilderVariable[_QBVC]] = {}

        for name, cls in params:
            if issubclass(cls, IRDLOperation):
                # Don't add the variables here, they will be added as they are traversed
                var = OperationVariable(name)
                qbvc = _IRDLOperationQBVC(var, self, {}, cls)
                fake_vars[name] = _QueryBuilderVariable(qbvc)

        # The root is given every time, so we add it type check immediately
        fake_vars["root"].qbvc__.register_var()

        func(**fake_vars)  # pyright: ignore[reportGeneralTypeIssues]

    def rewrite(
        self, func: Callable[Concatenate[PatternRewriter, QueryParams], None]
    ) -> QueryRewritePattern:
        def rewrite(match: Match, rewriter: PatternRewriter) -> None:
            return func(rewriter, **match)  # pyright: ignore[reportGeneralTypeIssues]

        return QueryRewritePattern(self, rewrite)
