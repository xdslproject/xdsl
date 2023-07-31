from __future__ import annotations

import abc
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
    BinaryConstraint,
    EqConstraint,
    Match,
    OperationVariable,
    PropertyConstraint,
    Query,
    SSAValueVariable,
    TypeConstraint,
    UnaryConstraint,
    Variable,
)
from xdsl.rewriting.query_rewrite_pattern import QueryRewritePattern

_T = TypeVar("_T")
_VariableT = TypeVar("_VariableT", bound=Variable[Any])


@dataclass
class QueryBuilder:
    _query: Query
    var_id: int = 0

    def next_var_id(self) -> str:
        id = self.var_id
        self.var_id = id + 1
        return f"{id}"

    @property
    def variables(self) -> dict[str, Variable[Any]]:
        return self._query.variables

    def add_variable(self, var: Variable[Any]) -> None:
        self._query.add_variable(var)

    def new_variable_context(
        self, qbvc_cls: type[_QBVCT], var_cls: type[Variable[Any]]
    ) -> _QBVCT:
        new_var = var_cls(self.next_var_id())
        new_qbvc = qbvc_cls(new_var, self, {})
        return new_qbvc

    def add_unary_constraint(self, constraint: UnaryConstraint[Any]):
        assert constraint.var.name in self._query.variables
        self._query.constraints.append(constraint)

    def add_binary_constraint(self, constraint: BinaryConstraint[Any, Any]):
        assert constraint.var0.name in self._query.variables
        assert constraint.var1.name in self._query.variables
        self._query.constraints.append(constraint)


@dataclass
class _QBVC(abc.ABC):
    """
    Query builder variable contents
    """

    var: Variable[Any]
    builder: QueryBuilder
    property_variables: dict[str, _QueryBuilderVariable[_QBVC]]

    def register_var(self) -> bool:
        """
        Returns False if the variable was already registered.
        """
        if self.var.name not in self.builder.variables:
            self.builder.add_variable(self.var)
        for var in self.property_variables.values():
            var.qbvc__.register_var()
        return True

    def constrain_type(self, hint: type[_T]) -> TypeGuard[_T]:
        if self.var.name not in self.builder.variables:
            self.builder.add_variable(self.var)
        self.builder.add_unary_constraint(TypeConstraint(self.var, hint))
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
        assert self.var.name in self.builder.variables
        other_qbvc = other_variable.qbvc__
        if other_qbvc.var.name not in self.builder.variables:
            self.builder.add_variable(other_qbvc.var)
        self.builder.add_binary_constraint(EqConstraint(self.var, other_qbvc.var))
        other_qbvc.register_var()
        return True

    def eq_value(self, self_variable: _QueryBuilderVariable[_QBVC], value: Any) -> bool:
        return False

    def get_attribute(self, __name: str) -> _QueryBuilderVariable[_QBVC]:
        attr = self.property_variables.get(__name)
        if attr is None:
            spec = self.attribute_spec(__name)
            if spec is None:
                raise AttributeError
            qbvc_cls, var_cls = spec
            attr = self._create_attribute_variable(__name, qbvc_cls, var_cls)
            # register property in this variable's cache
            self.property_variables[__name] = attr

        return attr

    def _create_attribute_variable(
        self, name: str, qbvc_cls: type[_QBVCT], var_cls: type[Variable[Any]]
    ) -> _QueryBuilderVariable[_QBVC]:
        new_qbvc = self.builder.new_variable_context(qbvc_cls, var_cls)
        new_var = new_qbvc.var
        self.builder.add_variable(new_var)
        self.builder.add_binary_constraint(PropertyConstraint(self.var, new_var, name))
        return _QueryBuilderVariable(new_qbvc)

    @abc.abstractmethod
    def attribute_spec(
        self, name: str
    ) -> tuple[type[_QBVC], type[Variable[Any]]] | None:
        raise NotImplementedError()


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
            return qbvc.get_attribute(__name)

    def __eq__(self, __value: object) -> bool:
        return self.qbvc__.eq(self, __value)


@dataclass
class _OperationQBVC(_QBVC):
    def attribute_spec(
        self, name: str
    ) -> tuple[type[_QBVC], type[Variable[Any]]] | None:
        raise NotImplementedError(f"TODO, {name}")


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
            self.builder.add_unary_constraint(TypeConstraint(self.var, self.cls))
        return did_register

    def attribute_spec(
        self, name: str
    ) -> tuple[type[_QBVC], type[Variable[Any]]] | None:
        if name in dict(self.op_def.operands):
            return _SSAValueQBVC, SSAValueVariable
        elif name in dict(self.op_def.results):
            raise NotImplementedError(f"TODO, {name}")
        elif name in dict(self.op_def.attributes):
            return _AttributeQBVC, AttributeVariable
        else:
            return None


class _SSAValueQBVC(_QBVC):
    var: SSAValueVariable

    def attribute_spec(
        self, name: str
    ) -> tuple[type[_QBVC], type[Variable[Any]]] | None:
        return _OperationQBVC, OperationVariable


class _AttributeQBVC(_QBVC):
    var: AttributeVariable

    def eq_value(self, self_variable: _QueryBuilderVariable[_QBVC], value: Any) -> bool:
        assert isinstance(value, Attribute)
        self.builder.add_unary_constraint(AttributeValueConstraint(self.var, value))
        return True

    def attribute_spec(
        self, name: str
    ) -> tuple[type[_QBVC], type[Variable[Any]]] | None:
        raise NotImplementedError(f"TODO, {name}")


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
        builder = QueryBuilder(self)

        for name, cls in params:
            if issubclass(cls, IRDLOperation):
                # Don't add the variables here, they will be added as they are traversed
                var = OperationVariable(name)
                qbvc = _IRDLOperationQBVC(var, builder, {}, cls)
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
