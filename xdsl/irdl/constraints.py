from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeGuard,
    cast,
)

from typing_extensions import TypeVar, deprecated

from xdsl.ir import (
    Attribute,
    AttributeCovT,
    ParametrizedAttribute,
    TypedAttribute,
)
from xdsl.utils.exceptions import PyRDLError, VerifyException
from xdsl.utils.runtime_final import is_runtime_final

if TYPE_CHECKING:
    from xdsl.irdl import IRDLAttrConstraint


@dataclass
class ConstraintContext:
    """
    Contains the assignment of constraint variables.
    """

    _variables: dict[str, Attribute] = field(default_factory=dict[str, Attribute])
    """The assignment of constraint variables."""

    _range_variables: dict[str, tuple[Attribute, ...]] = field(
        default_factory=dict[str, tuple[Attribute, ...]]
    )
    """The assignment of constraint range variables."""

    _int_variables: dict[str, int] = field(default_factory=dict[str, int])
    """The assignment of constraint int variables."""

    def get_variable(self, key: str) -> Attribute | None:
        return self._variables.get(key)

    def get_range_variable(self, key: str) -> tuple[Attribute, ...] | None:
        return self._range_variables.get(key)

    def get_int_variable(self, key: str) -> int | None:
        return self._int_variables.get(key)

    def set_attr_variable(self, key: str, attr: Attribute):
        self._variables[key] = attr

    def set_range_variable(self, key: str, attrs: tuple[Attribute, ...]):
        self._range_variables[key] = attrs

    def set_int_variable(self, key: str, i: int):
        self._int_variables[key] = i

    @property
    def attr_variables(self) -> AbstractSet[str]:
        return self._variables.keys()

    @property
    def range_variables(self) -> AbstractSet[str]:
        return self._range_variables.keys()

    @property
    def int_variables(self) -> AbstractSet[str]:
        return self._int_variables.keys()

    @deprecated("ConstraintContexts should not be copied")
    def copy(self):
        return ConstraintContext(
            self._variables.copy(),
            self._range_variables.copy(),
            self._int_variables.copy(),
        )

    @deprecated("ConstraintContexts should only be updated by set_* methods")
    def update(self, other: ConstraintContext):
        self._variables.update(other._variables)
        self._range_variables.update(other._range_variables)
        self._int_variables.update(other._int_variables)


_AttributeCovT = TypeVar(
    "_AttributeCovT", bound=Attribute, default=Attribute, covariant=True
)

ConstraintVariableType: TypeAlias = Attribute | Sequence[Attribute] | int
"""
Possible types that a constraint variable can have.
"""


@dataclass(frozen=True)
class AttrConstraint(ABC, Generic[AttributeCovT]):
    """Constrain an attribute to a certain value."""

    @abstractmethod
    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        """
        Check if the attribute satisfies the constraint,
        or raise an exception otherwise.
        """
        ...

    def verifies(self, attr: Attribute) -> TypeGuard[AttributeCovT]:
        """
        A helper method to check whether a given attribute matches `self`.
        """
        try:
            self.verify(attr, ConstraintContext())
            return True
        except VerifyException:
            return False

    def variables(self) -> set[str]:
        """
        Returns a set of the variables that can be extracted by this constraint.
        These variables are always expected to be set after running `verify`.
        """
        return set()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        """
        Check if there is enough information to infer the attribute given the
        constraint variables that are already set.
        """
        # By default, we cannot infer anything.
        return False

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        """
        Infer the attribute given the the values for all variables.

        Raises an exception if the attribute cannot be inferred. If `can_infer`
        returns `True` with the given constraint variables, this method should
        not raise an exception.
        """
        raise ValueError(f"Cannot infer attribute from constraint {self}")

    def get_bases(self) -> set[type[Attribute]] | None:
        """
        Get a set of base types that can satisfy this constraint, if there exists
        a finite collection, or None otherwise.
        """
        return None

    def __or__(
        self, value: AttrConstraint[_AttributeCovT], /
    ) -> AttrConstraint[AttributeCovT | _AttributeCovT]:
        if isinstance(value, AnyAttr) or self == value:
            return value  # pyright: ignore[reportReturnType]
        return AnyOf((self, value))

    def __and__(self, value: AttrConstraint, /) -> AttrConstraint[AttributeCovT]:
        if isinstance(value, AnyAttr) or self == value:
            return self
        return AllOf((self, value))  # pyright: ignore[reportReturnType]

    @abstractmethod
    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[AttributeCovT]:
        """
        A helper function to make type vars used in attribute definitions concrete when
        creating constraints for new attributes or operations.
        """
        raise NotImplementedError(
            "Custom constraints must map type vars in nested constraints, if any."
        )


ConstraintVariableTypeT = TypeVar(
    "ConstraintVariableTypeT", bound=ConstraintVariableType
)


TypedAttributeCovT = TypeVar("TypedAttributeCovT", bound=TypedAttribute, covariant=True)
TypedAttributeT = TypeVar("TypedAttributeT", bound=TypedAttribute)


@deprecated("Please use appropriate `AnyOf` constraints instead.")
@dataclass(frozen=True)
class TypedAttributeConstraint(AttrConstraint[TypedAttributeCovT]):
    """
    Constrains the type of a typed attribute.
    """

    attr_constraint: AttrConstraint[TypedAttributeCovT]
    type_constraint: AttrConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, TypedAttribute):
            raise VerifyException(f"attribute {attr} expected to be a TypedAttribute")
        self.attr_constraint.verify(attr, constraint_context)
        self.type_constraint.verify(attr.get_type(), constraint_context)

    def variables(self) -> set[str]:
        return self.type_constraint.variables() | self.attr_constraint.variables()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.attr_constraint.can_infer(var_constraint_names)

    def infer(self, context: ConstraintContext) -> TypedAttributeCovT:
        return self.attr_constraint.infer(context)

    def get_bases(self) -> set[type[Attribute]] | None:
        return self.attr_constraint.get_bases()

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> TypedAttributeConstraint[TypedAttributeCovT]:  # pyright: ignore[reportDeprecated]
        return TypedAttributeConstraint(  # pyright: ignore[reportDeprecated]
            self.attr_constraint.mapping_type_vars(type_var_mapping),
            self.type_constraint.mapping_type_vars(type_var_mapping),
        )


@dataclass(frozen=True)
class VarConstraint(AttrConstraint[AttributeCovT]):
    """
    Constrain an attribute with the given constraint, and constrain all occurences
    of this constraint (i.e, sharing the same name) to be equal.
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""

    constraint: AttrConstraint[AttributeCovT]
    """The constraint that the variable must satisfy."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        ctx_attr = constraint_context.get_variable(self.name)
        if ctx_attr is not None:
            if attr != ctx_attr:
                raise VerifyException(
                    f"attribute {constraint_context.get_variable(self.name)} expected from variable "
                    f"'{self.name}', but got {attr}"
                )
        else:
            self.constraint.verify(attr, constraint_context)
            constraint_context.set_attr_variable(self.name, attr)

    def variables(self) -> set[str]:
        return self.constraint.variables() | {self.name}

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        v = context.get_variable(self.name)
        return cast(AttributeCovT, v)

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.name in var_constraint_names

    def get_bases(self) -> set[type[Attribute]] | None:
        return self.constraint.get_bases()

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> VarConstraint[AttributeCovT]:
        return VarConstraint(
            self.name, self.constraint.mapping_type_vars(type_var_mapping)
        )


@dataclass(frozen=True)
class TypeVarConstraint(AttrConstraint):
    """
    Stores the TypeVar instance used to define a generic ParametrizedAttribute.
    """

    type_var: TypeVar
    """The instance of the TypeVar used in the definition."""

    base_constraint: AttrConstraint
    """Constraint inferred from the base of the TypeVar."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        self.base_constraint.verify(attr, constraint_context)

    def get_bases(self) -> set[type[Attribute]] | None:
        return self.base_constraint.get_bases()

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint:
        res = type_var_mapping.get(self.type_var)
        if res is None:
            raise KeyError(f"Mapping value missing for type var {self.type_var}")
        if not isinstance(res, AttrConstraint):
            raise ValueError(f"Unexpected constraint {res} for TypeVar {self.type_var}")
        return res


@dataclass(frozen=True, init=True)
class ConstraintVar:
    """
    Annotation used in PyRDL to define a constraint variable.
    For instance, the following code defines a constraint variable T,
    that can then be used in PyRDL:
    ```python
    T = Annotated[PyRDLConstraint, ConstraintVar("T")]
    ```
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""


@dataclass(frozen=True)
class EqAttrConstraint(AttrConstraint[AttributeCovT], Generic[AttributeCovT]):
    """Constrain an attribute to be equal to another attribute."""

    attr: AttributeCovT
    """The attribute we want to check equality with."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if attr != self.attr:
            raise VerifyException(f"Expected attribute {self.attr} but got {attr}")

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return True

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        return self.attr

    def get_bases(self) -> set[type[Attribute]] | None:
        return {type(self.attr)}

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[AttributeCovT]:
        return self


@dataclass(frozen=True)
class BaseAttr(AttrConstraint[AttributeCovT], Generic[AttributeCovT]):
    """Constrain an attribute to be of a given base type."""

    attr: type[AttributeCovT]
    """The expected attribute base type."""

    def __repr__(self):
        return f"BaseAttr({self.attr.__name__})"

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if not isinstance(attr, self.attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.attr.name}"
            )

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return (
            is_runtime_final(self.attr)
            and issubclass(self.attr, ParametrizedAttribute)
            and not self.attr.get_irdl_definition().parameters
        )

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        assert issubclass(self.attr, ParametrizedAttribute)
        attr = self.attr.new(())
        return attr

    def get_bases(self) -> set[type[Attribute]] | None:
        if is_runtime_final(self.attr):
            return {self.attr}
        return None

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[AttributeCovT]:
        return self


@deprecated("Please use `irdl_to_attr_constraint` instead")
def attr_constr_coercion(
    attr: AttributeCovT | type[AttributeCovT] | AttrConstraint[AttributeCovT],
) -> AttrConstraint[AttributeCovT]:
    """
    Attributes are coerced into EqAttrConstraints,
    and Attribute types are coerced into BaseAttr.
    """
    from xdsl.irdl import irdl_to_attr_constraint

    return irdl_to_attr_constraint(attr)


@dataclass(frozen=True)
class AnyAttr(AttrConstraint):
    """Constraint that is verified by all attributes."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        pass

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AnyAttr:
        return self

    def __or__(self, value: AttrConstraint[_AttributeCovT], /):
        return self

    def __and__(self, value: AttrConstraint[AttributeCovT], /):
        return value


@dataclass(frozen=True, init=False)
class AnyOf(AttrConstraint[AttributeCovT], Generic[AttributeCovT]):
    """Ensure that an attribute satisfies one of the given constraints."""

    attr_constrs: tuple[AttrConstraint[AttributeCovT], ...]
    """
    The tuple of attribute constraints that are checked by this `AnyOf` constraint.

    At most one constraint may be "abstract": an abstract constraint is a `BaseAttr`
    for an abstract base class which can be inherited by other attribute classes.
    This abstract attribute type is not runtime-final (i.e., does not have the `@irdl_attr_definition` decorator).
    """

    _eq_constrs: set[Attribute] = field(hash=False, repr=False)
    _based_constrs: dict[type[Attribute], AttrConstraint[AttributeCovT]] = field(
        hash=False, repr=False
    )
    _abstr_constr: AttrConstraint[AttributeCovT] | None = field(hash=False, repr=False)

    def __init__(
        self,
        attr_constrs: Sequence[
            AttributeCovT | type[AttributeCovT] | AttrConstraint[AttributeCovT]
        ],
    ):
        from xdsl.irdl import irdl_to_attr_constraint

        constrs: tuple[AttrConstraint[AttributeCovT], ...] = tuple(
            irdl_to_attr_constraint(constr) for constr in attr_constrs
        )

        eq_constrs = set[Attribute]()
        based_constrs = dict[type[Attribute], AttrConstraint[AttributeCovT]]()

        bases = set[type[Attribute]]()
        eq_bases = set[type[Attribute]]()
        abstr_constr: AttrConstraint[AttributeCovT] | None = None
        for i, c in enumerate(constrs):
            b = c.get_bases()
            if b is None:
                if abstr_constr is not None:
                    raise PyRDLError(
                        f"Cannot form `AnyOf` constraint with both {c} and {abstr_constr}, "
                        "as they cannot be verified as disjoint."
                    )
                if not isinstance(c, BaseAttr) or is_runtime_final(c.attr):
                    raise PyRDLError(
                        f"Constraint in `AnyOf` without bases must be a `BaseAttr` "
                        f"of a non-final abstract attribute class, got {c} instead."
                    )
                abstr_constr = c
                continue

            if not b.isdisjoint(bases):
                raise PyRDLError(
                    f"Constraint {c} shares a base with a non-equality constraint "
                    f"in {set(constrs[0:i])} in `AnyOf` constraint."
                )

            if isinstance(c, EqAttrConstraint):
                eq_constrs.add(c.attr)
                eq_bases |= b
            else:
                if not b.isdisjoint(eq_bases):
                    raise PyRDLError(
                        f"Non-equality constraint {c} shares a base with a constraint "
                        f"in {set(constrs[0:i])} in `AnyOf` constraint."
                    )
                for base in b:
                    based_constrs[base] = c
                bases |= b

        # check for overlaps with the abstract constraint
        if abstr_constr is not None:
            # equality constraints should not overlap
            for attr in eq_constrs:
                if isinstance(attr, abstr_constr.attr):
                    raise PyRDLError(
                        f"Equality constraint {EqAttrConstraint(attr)} overlaps with the "
                        f"constraint {abstr_constr} in `AnyOf` constraint."
                    )
            # bases should not overlap via issubclass
            for base in bases:
                if issubclass(base, abstr_constr.attr):
                    raise PyRDLError(
                        f"Non-equality constraint {based_constrs[base]} overlaps with "
                        f"the constraint {abstr_constr} in `AnyOf` constraint."
                    )

        object.__setattr__(
            self,
            "attr_constrs",
            constrs,
        )
        object.__setattr__(
            self,
            "_eq_constrs",
            eq_constrs,
        )
        object.__setattr__(
            self,
            "_based_constrs",
            based_constrs,
        )
        object.__setattr__(self, "_abstr_constr", abstr_constr)

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if attr in self._eq_constrs:
            return
        constr = self._based_constrs.get(attr.__class__)
        if constr is not None:
            constr.verify(attr, constraint_context)
            return
        # Try abstract constraint if present
        if self._abstr_constr is not None:
            self._abstr_constr.verify(attr, constraint_context)
            return
        raise VerifyException(f"Unexpected attribute {attr}")

    def __or__(
        self, value: AttrConstraint[_AttributeCovT], /
    ) -> AnyOf[AttributeCovT | _AttributeCovT]:
        return AnyOf((*self.attr_constrs, value))

    def variables(self) -> set[str]:
        if not self.attr_constrs:
            return set()
        variables = self.attr_constrs[0].variables()
        for constr in self.attr_constrs[1:]:
            variables &= constr.variables()
        return variables

    def get_bases(self) -> set[type[Attribute]] | None:
        bases = set[type[Attribute]]()
        for constr in self.attr_constrs:
            b = constr.get_bases()
            if b is None:
                return
            bases |= b
        return bases

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AnyOf[AttributeCovT]:
        return AnyOf(
            tuple(c.mapping_type_vars(type_var_mapping) for c in self.attr_constrs)
        )


@dataclass(frozen=True)
class AllOf(AttrConstraint[AttributeCovT]):
    """Ensure that an attribute satisfies all the given constraints."""

    attr_constrs: tuple[AttrConstraint[AttributeCovT], ...]
    """The list of constraints that are checked."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        exc_bucket: list[VerifyException] = []

        for attr_constr in self.attr_constrs:
            try:
                attr_constr.verify(attr, constraint_context)
            except VerifyException as e:
                exc_bucket.append(e)

        if len(exc_bucket):
            if len(exc_bucket) == 1:
                raise VerifyException(str(exc_bucket[0])) from exc_bucket[0]
            exc_msg = "The following constraints were not satisfied:\n"
            exc_msg += "\n".join([str(e) for e in exc_bucket])
            raise VerifyException(exc_msg)

    def variables(self) -> set[str]:
        vars = set[str]()
        for constr in self.attr_constrs:
            vars |= constr.variables()
        return vars

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return any(
            constr.can_infer(var_constraint_names) for constr in self.attr_constrs
        )

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        for constr in self.attr_constrs:
            if constr.can_infer(context.attr_variables):
                return constr.infer(context)
        raise ValueError("Cannot infer attribute from constraint")

    def get_bases(self) -> set[type[Attribute]] | None:
        bases: set[type[Attribute]] | None = None
        for constr in self.attr_constrs:
            b = constr.get_bases()
            if b is None:
                continue
            if bases is None:
                bases = b
            else:
                bases &= b
        return bases

    def __and__(self, value: AttrConstraint, /) -> AllOf[AttributeCovT]:
        return AllOf((*self.attr_constrs, value))  # pyright: ignore[reportReturnType]

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AllOf[AttributeCovT]:
        return AllOf(
            tuple(c.mapping_type_vars(type_var_mapping) for c in self.attr_constrs)
        )


ParametrizedAttributeT = TypeVar("ParametrizedAttributeT", bound=ParametrizedAttribute)
ParametrizedAttributeCovT = TypeVar(
    "ParametrizedAttributeCovT", bound=ParametrizedAttribute, covariant=True
)


@dataclass(frozen=True, init=False)
class ParamAttrConstraint(
    AttrConstraint[ParametrizedAttributeCovT], Generic[ParametrizedAttributeCovT]
):
    """
    Constrain an attribute to be of a given type,
    and also constrain its parameters with additional constraints.
    """

    base_attr: type[ParametrizedAttributeCovT]
    """The base attribute type."""

    param_constrs: tuple[AttrConstraint, ...]
    """The attribute parameter constraints"""

    def __init__(
        self,
        base_attr: type[ParametrizedAttributeCovT],
        param_constrs: Sequence[IRDLAttrConstraint | None],
    ):
        from xdsl.irdl import irdl_to_attr_constraint

        constrs = tuple(
            irdl_to_attr_constraint(constr) if constr is not None else AnyAttr()
            for constr in param_constrs
        )
        object.__setattr__(self, "base_attr", base_attr)
        object.__setattr__(self, "param_constrs", constrs)

    def __repr__(self):
        return f"ParamAttrConstraint({self.base_attr.__name__}, {repr(self.param_constrs)})"

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if not isinstance(attr, self.base_attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.base_attr.name}"
            )
        parameters = attr.parameters
        if len(self.param_constrs) != len(parameters):
            raise VerifyException(
                f"{len(self.param_constrs)} parameters expected, "
                f"but got {len(parameters)}"
            )
        for idx, param_constr in enumerate(self.param_constrs):
            param_constr.verify(parameters[idx], constraint_context)

    def variables(self) -> set[str]:
        vars = set[str]()
        for constr in self.param_constrs:
            vars |= constr.variables()
        return vars

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return is_runtime_final(self.base_attr) and all(
            constr.can_infer(var_constraint_names) for constr in self.param_constrs
        )

    def infer(self, context: ConstraintContext) -> ParametrizedAttributeCovT:
        params = tuple(constr.infer(context) for constr in self.param_constrs)
        attr = self.base_attr.new(params)
        return attr

    def get_bases(self) -> set[type[Attribute]] | None:
        if is_runtime_final(self.base_attr):
            return {self.base_attr}
        return None

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> ParamAttrConstraint[ParametrizedAttributeCovT]:
        return ParamAttrConstraint(
            self.base_attr,
            tuple(c.mapping_type_vars(type_var_mapping) for c in self.param_constrs),
        )

    def __or__(self, value: AttrConstraint[_AttributeCovT], /):
        if (
            not isinstance(value, ParamAttrConstraint)
            or self.base_attr is not cast(ParamAttrConstraint[Any], value).base_attr
        ):
            return super().__or__(value)  # pyright: ignore[reportUnknownArgumentType]
        return ParamAttrConstraint(
            self.base_attr,
            tuple(
                l | r
                for l, r in zip(self.param_constrs, value.param_constrs, strict=True)
            ),
        )


@dataclass(frozen=True, init=False)
class MessageConstraint(AttrConstraint[AttributeCovT]):
    """
    Attach a message to a constraint, to provide more context when the constraint
    is not satisfied.
    """

    constr: AttrConstraint[AttributeCovT]
    message: str

    def __init__(
        self,
        constr: (AttrConstraint[AttributeCovT] | AttributeCovT | type[AttributeCovT]),
        message: str,
    ):
        from xdsl.irdl import irdl_to_attr_constraint

        object.__setattr__(self, "constr", irdl_to_attr_constraint(constr))
        object.__setattr__(self, "message", message)

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        try:
            return self.constr.verify(attr, constraint_context)
        except VerifyException as e:
            raise VerifyException(
                f"{self.message}\nUnderlying verification failure: {e.args[0]}",
                *e.args[1:],
            )

    def variables(self) -> set[str]:
        return self.constr.variables()

    def get_bases(self) -> set[type[Attribute]] | None:
        return self.constr.get_bases()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.constr.can_infer(var_constraint_names)

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        return self.constr.infer(context)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> MessageConstraint[AttributeCovT]:
        return MessageConstraint(
            self.constr.mapping_type_vars(type_var_mapping), self.message
        )


@dataclass(frozen=True)
class IntConstraint(ABC):
    """Constrain an integer to certain values."""

    @abstractmethod
    def verify(
        self,
        i: int,
        constraint_context: ConstraintContext,
    ) -> None:
        """
        Check if the integer satisfies the constraint, or raise an exception otherwise.
        """
        ...

    def variables(self) -> set[str]:
        """
        Returns a set of the variables that can be extracted by this constraint.
        These variables are always expected to be set after running `verify`.
        """
        return set()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        """
        Check if there is enough information to infer the integer given the
        constraint variables that are already set.
        """
        # By default, we cannot infer anything.
        return False

    def infer(self, context: ConstraintContext) -> int:
        """
        Infer the attribute given the the values for all variables.

        Raises an exception if the attribute cannot be inferred. If `can_infer`
        returns `True` with the given constraint variables, this method should
        not raise an exception.
        """
        raise ValueError(f"Cannot infer integer from constraint {self}")

    @abstractmethod
    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        """
        A helper function to make type vars used in attribute definitions concrete when
        creating constraints for new attributes or operations.
        """
        raise NotImplementedError(
            "Custom constraints must map type vars in nested constraints, if any."
        )


class AnyInt(IntConstraint):
    """
    Constraint that is verified by all integers.
    """

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        pass

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return self


@dataclass(frozen=True)
class EqIntConstraint(IntConstraint):
    """Constrain an integer to a value."""

    value: int

    def verify(
        self,
        i: int,
        constraint_context: ConstraintContext,
    ) -> None:
        if self.value != i:
            raise VerifyException(f"Invalid value {i}, expected {self.value}")

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return True

    def infer(self, context: ConstraintContext) -> int:
        return self.value

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return self


@dataclass(frozen=True)
class NotEqualIntConstraint(IntConstraint):
    """Constrain an integer to not be equal to a given value."""

    value: int
    """The value the integer must not be equal to."""

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        if i == self.value:
            raise VerifyException(f"expected integer != {self.value}")

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return self


@dataclass(frozen=True)
class IntSetConstraint(IntConstraint):
    """Constrain an integer to one of a set of integers."""

    values: frozenset[int]

    def verify(
        self,
        i: int,
        constraint_context: ConstraintContext,
    ) -> None:
        if i not in self.values:
            set_str = set(self.values) if self.values else "{}"
            raise VerifyException(f"Invalid value {i}, expected one of {set_str}")

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return len(self.values) == 1

    def infer(self, context: ConstraintContext) -> int:
        return next(iter(self.values))

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return self


@dataclass(frozen=True)
class AtLeast(IntConstraint):
    """Constrain an integer to be at least a given value."""

    bound: int
    """The minimum value the integer can take."""

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        if i < self.bound:
            raise VerifyException(f"expected integer >= {self.bound}, got {i}")

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return self


@dataclass(frozen=True)
class AtMost(IntConstraint):
    """Constrain an integer to be at most a given value."""

    bound: int
    """The maximum value the integer can take."""

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        if i > self.bound:
            raise VerifyException(f"Expected integer <= {self.bound}, got {i}")

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return self


@dataclass(frozen=True)
class IntVarConstraint(IntConstraint):
    """
    Constrain an integer with the given constraint, and constrain all occurences
    of this constraint (i.e, sharing the same name) to be equal.
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""

    constraint: IntConstraint
    """The constraint that the variable must satisfy."""

    def verify(
        self,
        i: int,
        constraint_context: ConstraintContext,
    ) -> None:
        if self.name in constraint_context.int_variables:
            if i != constraint_context.get_int_variable(self.name):
                raise VerifyException(
                    f"integer {constraint_context.get_int_variable(self.name)} expected from int variable "
                    f"'{self.name}', but got {i}"
                )
        else:
            self.constraint.verify(i, constraint_context)
            constraint_context.set_int_variable(self.name, i)

    def variables(self) -> set[str]:
        return self.constraint.variables() | {self.name}

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.name in var_constraint_names

    def infer(
        self,
        context: ConstraintContext,
    ) -> int:
        v = context.get_int_variable(self.name)
        assert isinstance(v, int)
        return v

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return IntVarConstraint(
            self.name, self.constraint.mapping_type_vars(type_var_mapping)
        )


@dataclass(frozen=True)
class IntTypeVarConstraint(IntConstraint):
    """
    Stores the TypeVar instance used to define a generic type.
    """

    type_var: TypeVar
    """The instance of the TypeVar used in the definition."""

    base_constraint: IntConstraint
    """Constraint inferred from the base of the TypeVar."""

    def verify(
        self,
        i: int,
        constraint_context: ConstraintContext,
    ) -> None:
        self.base_constraint.verify(i, constraint_context)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        res = type_var_mapping.get(self.type_var)
        if res is None:
            raise KeyError(f"Mapping value missing for type var {self.type_var}")
        if not isinstance(res, IntConstraint):
            raise ValueError(f"Unexpected constraint {res} for TypeVar {self.type_var}")
        return res


@dataclass(frozen=True)
class RangeConstraint(ABC, Generic[AttributeCovT]):
    """Constrain a range of attributes to certain values."""

    @abstractmethod
    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        """
        Check if the range satisfies the constraint, or raise an exception otherwise.
        """
        ...

    def verifies(
        self,
        attrs: Sequence[Attribute],
    ) -> TypeGuard[AttributeCovT]:
        """
        A helper method to check whether a given attribute matches `self`.
        """
        try:
            self.verify(attrs, ConstraintContext())
            return True
        except VerifyException:
            return False

    @abstractmethod
    def verify_length(self, length: int, constraint_context: ConstraintContext) -> None:
        """
        Check if the length of the range satisfies the constraint, or raise an exception otherwise.
        """
        ...

    def variables(self) -> set[str]:
        """
        Returns a set of the variables that can be extracted by this constraint.
        These variables are always expected to be set after running `verify`.
        """
        return set()

    def variables_from_length(self) -> set[str]:
        """
        Returns a set of the variables that can be extracted from the range length by this constraint.
        These variables are always expected to be set after running `verify_length`.
        """
        return set()

    def can_infer(
        self, var_constraint_names: AbstractSet[str], *, length_known: bool
    ) -> bool:
        """
        Check if there is enough information to infer the attribute given the
        constraint variables that are already set, and whether the length of the
        range is known in advance.
        """
        # By default, we cannot infer anything.
        return False

    def infer(
        self, context: ConstraintContext, *, length: int | None
    ) -> Sequence[AttributeCovT]:
        """
        Infer the attribute given the the values for all variables, and possibly
        the length of the range if known.

        Raises an exception if the attribute cannot be inferred. If `can_infer`
        returns `True` with the given constraint variables, this method should
        not raise an exception.
        """
        raise ValueError(f"Cannot infer range from constraint {self}")

    @abstractmethod
    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> RangeConstraint[AttributeCovT]:
        """
        A helper function to make type vars used in attribute definitions concrete when
        creating constraints for new attributes or operations.
        """
        raise NotImplementedError(
            "Custom constraints must map type vars in nested constraints, if any."
        )

    def of_length(
        self, length_constr: IntConstraint
    ) -> RangeLengthConstraint[AttributeCovT]:
        return RangeLengthConstraint(self, length_constr)


@dataclass(frozen=True)
class RangeLengthConstraint(RangeConstraint[AttributeCovT]):
    """
    Constrain an attribute range with the given length.
    """

    constraint: RangeConstraint[AttributeCovT]
    """The constraint that the variable must satisfy."""

    length: IntConstraint
    """The length that the range must have"""

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        self.verify_length(len(attrs), constraint_context)
        self.constraint.verify(attrs, constraint_context)

    def verify_length(self, length: int, constraint_context: ConstraintContext) -> None:
        try:
            self.length.verify(length, constraint_context)
        except VerifyException as e:
            raise VerifyException(
                "incorrect length for range variable:\n" + str(e)
            ) from e

    def variables(self) -> set[str]:
        return self.constraint.variables() | self.length.variables()

    def variables_from_length(self) -> set[str]:
        return self.length.variables()

    def can_infer(
        self, var_constraint_names: AbstractSet[str], *, length_known: bool
    ) -> bool:
        length_known = length_known or self.length.can_infer(var_constraint_names)
        return self.constraint.can_infer(
            var_constraint_names, length_known=length_known
        )

    def infer(
        self, context: ConstraintContext, *, length: int | None
    ) -> Sequence[AttributeCovT]:
        if length is None:
            length = self.length.infer(context)
        return self.constraint.infer(context, length=length)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> RangeLengthConstraint[AttributeCovT]:
        return RangeLengthConstraint(
            self.constraint.mapping_type_vars(type_var_mapping),
            self.length.mapping_type_vars(type_var_mapping),
        )


@dataclass(frozen=True)
class RangeVarConstraint(RangeConstraint[AttributeCovT]):
    """
    Constrain an attribute range with the given constraint, and constrain all occurences
    of this constraint (i.e, sharing the same name) to be equal.
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""

    constraint: RangeConstraint[AttributeCovT]
    """The constraint that the variable must satisfy."""

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        ctx_attrs = constraint_context.get_range_variable(self.name)
        if ctx_attrs is not None:
            if attrs != ctx_attrs:
                raise VerifyException(
                    f"attributes {tuple(str(x) for x in ctx_attrs)} expected from range variable "
                    f"'{self.name}', but got {tuple(str(x) for x in attrs)}"
                )
        else:
            self.constraint.verify(attrs, constraint_context)
            constraint_context.set_range_variable(self.name, tuple(attrs))

    def verify_length(self, length: int, constraint_context: ConstraintContext) -> None:
        # It is not possible to fully verify the constraint from just the length, so we don't try.
        pass

    def variables(self) -> set[str]:
        return self.constraint.variables() | {self.name}

    def can_infer(
        self, var_constraint_names: AbstractSet[str], *, length_known: bool
    ) -> bool:
        return self.name in var_constraint_names

    def infer(
        self, context: ConstraintContext, *, length: int | None
    ) -> Sequence[AttributeCovT]:
        v = context.get_range_variable(self.name)
        return cast(Sequence[AttributeCovT], v)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> RangeVarConstraint[AttributeCovT]:
        return RangeVarConstraint(
            self.name, self.constraint.mapping_type_vars(type_var_mapping)
        )


@dataclass(frozen=True)
class RangeOf(RangeConstraint[AttributeCovT]):
    """
    Constrain each element in a range to satisfy a given constraint.
    """

    constr: AttrConstraint[AttributeCovT]

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        for a in attrs:
            self.constr.verify(a, constraint_context)

    def verify_length(self, length: int, constraint_context: ConstraintContext): ...

    def variables(self) -> set[str]:
        return self.constr.variables()

    def can_infer(
        self, var_constraint_names: AbstractSet[str], *, length_known: bool
    ) -> bool:
        return length_known and self.constr.can_infer(var_constraint_names)

    def infer(
        self,
        context: ConstraintContext,
        *,
        length: int | None,
    ) -> Sequence[AttributeCovT]:
        assert length is not None
        attr = self.constr.infer(context)
        return (attr,) * length

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> RangeOf[AttributeCovT]:
        return RangeOf(self.constr.mapping_type_vars(type_var_mapping))


@dataclass(frozen=True)
class SingleOf(RangeConstraint[AttributeCovT]):
    """
    Constrain a range to only contain a single element, which should satisfy a given constraint.
    """

    constr: AttrConstraint[AttributeCovT]

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        if len(attrs) != 1:
            raise VerifyException(f"Expected a single attribute, got {len(attrs)}")
        self.constr.verify(attrs[0], constraint_context)

    def verify_length(self, length: int, constraint_context: ConstraintContext) -> None:
        if length != 1:
            raise VerifyException(f"Expected a single attribute, got {length}")

    def variables(self) -> set[str]:
        return self.constr.variables()

    def can_infer(
        self, var_constraint_names: AbstractSet[str], *, length_known: int | None
    ) -> bool:
        return self.constr.can_infer(var_constraint_names)

    def infer(
        self, context: ConstraintContext, *, length: int | None
    ) -> Sequence[AttributeCovT]:
        return (self.constr.infer(context),)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> SingleOf[AttributeCovT]:
        return SingleOf(self.constr.mapping_type_vars(type_var_mapping))
