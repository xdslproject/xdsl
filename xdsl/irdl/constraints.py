from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from inspect import isclass
from typing import Generic, TypeAlias, TypeVar

from xdsl.ir import Attribute, AttributeCovT, ParametrizedAttribute
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.runtime_final import is_runtime_final


@dataclass
class ConstraintContext:
    """
    Contains the assignment of constraint variables.
    """

    variables: dict[str, Attribute] = field(default_factory=dict)
    """The assignment of constraint variables."""

    def copy(self):
        return ConstraintContext(self.variables.copy())

    def update(self, other: ConstraintContext):
        self.variables.update(other.variables)


_AttributeCovT = TypeVar("_AttributeCovT", bound=Attribute, covariant=True)


@dataclass(frozen=True)
class GenericAttrConstraint(Generic[AttributeCovT], ABC):
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

    def get_resolved_variables(self) -> set[str]:
        """
        Get the set of type variables that are always resolved when verifying
        the constraint.
        """
        return set()

    def can_infer(self, constraint_names: set[str]) -> bool:
        """
        Check if there is enough information to infer the attribute given the
        constraint variables that are already set.
        """
        # By default, we cannot infer anything.
        return False

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        """
        Infer the attribute given the constraint variables that are already set.

        Raises an exception if the attribute cannot be inferred. If `can_infer`
        returns `True` with the given constraint variables, this method should
        not raise an exception.
        """
        raise ValueError("Cannot infer attribute from constraint")

    def get_unique_base(self) -> type[Attribute] | None:
        """Get the unique base type that can satisfy the constraint, if any."""
        return None

    def __or__(
        self, value: GenericAttrConstraint[_AttributeCovT], /
    ) -> AnyOf[AttributeCovT | _AttributeCovT]:
        return AnyOf((self, value))


AttrConstraint: TypeAlias = GenericAttrConstraint[Attribute]


@dataclass(frozen=True)
class VarConstraint(GenericAttrConstraint[AttributeCovT]):
    """
    Constraint variable. If the variable is already set, this will constrain
    the attribute to be equal to the variable. Otherwise, it will first check that the
    variable satisfies the variable constraint, then set the variable with the
    attribute.
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""

    constraint: GenericAttrConstraint[AttributeCovT]
    """The constraint that the variable must satisfy."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if self.name in constraint_context.variables:
            if attr != constraint_context.variables[self.name]:
                raise VerifyException(
                    f"attribute {constraint_context.variables[self.name]} expected from variable "
                    f"'{self.name}', but got {attr}"
                )
        else:
            self.constraint.verify(attr, constraint_context)
            constraint_context.variables[self.name] = attr

    def get_resolved_variables(self) -> set[str]:
        return {self.name, *self.constraint.get_resolved_variables()}

    def can_infer(self, constraint_names: set[str]) -> bool:
        return self.name in constraint_names

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        constraint_context = constraint_context or ConstraintContext()
        if self.name not in constraint_context.variables:
            raise ValueError(f"Cannot infer attribute from constraint {self}")
        return constraint_context.variables[self.name]

    def get_unique_base(self) -> type[Attribute] | None:
        return self.constraint.get_unique_base()


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
class EqAttrConstraint(Generic[AttributeCovT], GenericAttrConstraint[AttributeCovT]):
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

    def can_infer(self, constraint_names: set[str]) -> bool:
        return True

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        return self.attr

    def get_unique_base(self) -> type[Attribute] | None:
        return type(self.attr)


@dataclass(frozen=True)
class BaseAttr(Generic[AttributeCovT], GenericAttrConstraint[AttributeCovT]):
    """Constrain an attribute to be of a given base type."""

    attr: type[AttributeCovT]
    """The expected attribute base type."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if not isinstance(attr, self.attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.attr.name}"
            )

    def get_unique_base(self) -> type[Attribute] | None:
        if is_runtime_final(self.attr):
            return self.attr
        return None


def attr_constr_coercion(
    attr: AttributeCovT | type[AttributeCovT] | GenericAttrConstraint[AttributeCovT],
) -> GenericAttrConstraint[AttributeCovT]:
    """
    Attributes are coerced into EqAttrConstraints,
    and Attribute types are coerced into BaseAttr.
    """
    if isinstance(attr, GenericAttrConstraint):
        return attr
    if isinstance(attr, Attribute):
        return EqAttrConstraint(attr)
    if isclass(attr):
        return BaseAttr(attr)
    assert False


@dataclass(frozen=True)
class AnyAttr(GenericAttrConstraint[Attribute]):
    """Constraint that is verified by all attributes."""

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        pass


@dataclass(frozen=True, init=False)
class AnyOf(Generic[AttributeCovT], GenericAttrConstraint[AttributeCovT]):
    """Ensure that an attribute satisfies one of the given constraints."""

    attr_constrs: tuple[GenericAttrConstraint[Attribute], ...]
    """The list of constraints that are checked."""

    def __init__(
        self,
        attr_constrs: Sequence[
            Attribute | type[Attribute] | GenericAttrConstraint[Attribute]
        ],
    ):
        constrs: tuple[GenericAttrConstraint[Attribute], ...] = tuple(
            attr_constr_coercion(constr) for constr in attr_constrs
        )
        object.__setattr__(
            self,
            "attr_constrs",
            constrs,
        )

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext | None = None,
    ) -> None:
        constraint_context = constraint_context or ConstraintContext()
        for attr_constr in self.attr_constrs:
            # Copy the constraint to ensure that if the constraint fails, the
            # constraint context is not modified.
            constraint_context_copy = constraint_context.copy()
            try:
                attr_constr.verify(attr, constraint_context_copy)
                # If the constraint succeeds, we update back the constraint variables
                constraint_context.update(constraint_context_copy)
                return
            except VerifyException:
                pass
        raise VerifyException(f"Unexpected attribute {attr}")

    def __or__(
        self, value: GenericAttrConstraint[_AttributeCovT], /
    ) -> AnyOf[AttributeCovT | _AttributeCovT]:
        return AnyOf((*self.attr_constrs, value))

    def get_resolved_variables(self) -> set[str]:
        if len(self.attr_constrs) == 0:
            return set()
        return set[str].intersection(
            *(constr.get_resolved_variables() for constr in self.attr_constrs)
        )

    def get_unique_base(self) -> type[Attribute] | None:
        bases = [constr.get_unique_base() for constr in self.attr_constrs]
        if None in bases:
            return None
        if len(set(bases)) == 1:
            return bases[0]
        return None


@dataclass(frozen=True)
class AllOf(GenericAttrConstraint[AttributeCovT]):
    """Ensure that an attribute satisfies all the given constraints."""

    attr_constrs: tuple[GenericAttrConstraint[AttributeCovT], ...]
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

    def get_resolved_variables(self) -> set[str]:
        if len(self.attr_constrs) == 0:
            return set()
        return set[str].union(
            *[constr.get_resolved_variables() for constr in self.attr_constrs]
        )

    def can_infer(self, constraint_names: set[str]) -> bool:
        return any(constr.can_infer(constraint_names) for constr in self.attr_constrs)

    def infer(self, constraint_context: ConstraintContext | None = None) -> Attribute:
        constraint_context = constraint_context or ConstraintContext()
        for constr in self.attr_constrs:
            if constr.can_infer(set(constraint_context.variables.keys())):
                return constr.infer(constraint_context)
        raise ValueError("Cannot infer attribute from constraint")

    def get_unique_base(self) -> type[Attribute] | None:
        # This could be improved if we keep track of all the possible base types for
        # each constraint.
        for constr in self.attr_constrs:
            base = constr.get_unique_base()
            if base is not None:
                return base
        return None


ParametrizedAttributeCovT = TypeVar(
    "ParametrizedAttributeCovT", bound=ParametrizedAttribute, covariant=True
)


@dataclass(frozen=True, init=False)
class ParamAttrConstraint(
    Generic[ParametrizedAttributeCovT], GenericAttrConstraint[ParametrizedAttributeCovT]
):
    """
    Constrain an attribute to be of a given type,
    and also constrain its parameters with additional constraints.
    """

    base_attr: type[ParametrizedAttributeCovT]
    """The base attribute type."""

    param_constrs: tuple[GenericAttrConstraint[Attribute], ...]
    """The attribute parameter constraints"""

    def __init__(
        self,
        base_attr: type[ParametrizedAttributeCovT],
        param_constrs: Sequence[
            (Attribute | type[Attribute] | GenericAttrConstraint[Attribute] | None)
        ],
    ):
        constrs = tuple(
            attr_constr_coercion(constr) if constr is not None else AnyAttr()
            for constr in param_constrs
        )
        object.__setattr__(self, "base_attr", base_attr)
        object.__setattr__(self, "param_constrs", constrs)

    def verify(
        self,
        attr: Attribute,
        constraint_context: ConstraintContext,
    ) -> None:
        if not isinstance(attr, self.base_attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.base_attr.name}"
            )
        if len(self.param_constrs) != len(attr.parameters):
            raise VerifyException(
                f"{len(self.param_constrs)} parameters expected, "
                f"but got {len(attr.parameters)}"
            )
        for idx, param_constr in enumerate(self.param_constrs):
            param_constr.verify(attr.parameters[idx], constraint_context)

    def get_resolved_variables(self) -> set[str]:
        if not self.param_constrs:
            return set()
        return {
            var
            for constr in self.param_constrs
            for var in constr.get_resolved_variables()
        }

    def get_unique_base(self) -> type[Attribute] | None:
        if is_runtime_final(self.base_attr):
            return self.base_attr
        return None


@dataclass(frozen=True, init=False)
class MessageConstraint(GenericAttrConstraint[AttributeCovT]):
    """
    Attach a message to a constraint, to provide more context when the constraint
    is not satisfied.
    """

    constr: GenericAttrConstraint[AttributeCovT]
    message: str

    def __init__(
        self,
        constr: (
            GenericAttrConstraint[AttributeCovT] | AttributeCovT | type[AttributeCovT]
        ),
        message: str,
    ):
        object.__setattr__(self, "constr", attr_constr_coercion(constr))
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

    def get_resolved_variables(self) -> set[str]:
        return self.constr.get_resolved_variables()

    def get_unique_base(self) -> type[Attribute] | None:
        return self.constr.get_unique_base()

    def can_infer(self, constraint_names: set[str]) -> bool:
        return self.constr.can_infer(constraint_names)

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        return self.constr.infer(constraint_context)


class RangeConstraint(Generic[AttributeCovT], ABC):
    @abstractmethod
    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        """
        Check if the range satisfies the constraint, or raise an exception otherwise.
        The range can contain Nones, which represent an attribute not to be checked.
        """
        ...

    def get_resolved_variables(self) -> set[str]:
        """
        Get the set of type variables that are always resolved when verifying
        the constraint.
        """
        return set()

    def can_infer(self, constraint_names: set[str]) -> bool:
        """
        Check if there is enough information to infer the range given the
        constraint variables that are already set.
        """
        # By default, we cannot infer anything.
        return False

    def infer(
        self, length: int, constraint_context: ConstraintContext
    ) -> list[Attribute]:
        """
        Infer the range given the constraint variables that are already set.

        Raises an exception if the range cannot be inferred. If `can_infer`
        returns `True` with the given constraint variables, this method should
        not raise an exception.
        """
        raise ValueError("Cannot infer range from constraint")


@dataclass
class RangeOf(RangeConstraint[AttributeCovT]):
    """
    Constrain each element in a range to satisfy a given constraint.
    """

    constr: GenericAttrConstraint[AttributeCovT]

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        for a in attrs:
            self.constr.verify(a, constraint_context)

    def get_resolved_variables(self) -> set[str]:
        return self.constr.get_resolved_variables()

    def can_infer(self, constraint_names: set[str]) -> bool:
        return self.constr.can_infer(constraint_names)

    def infer(
        self, length: int, constraint_context: ConstraintContext
    ) -> list[Attribute]:
        return [self.constr.infer(constraint_context)] * length


@dataclass
class SingleOf(RangeConstraint[AttributeCovT]):
    """
    Constrain a range to only contain a single element, which should satisfy a given constraint.
    """

    constr: GenericAttrConstraint[AttributeCovT]

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        if len(attrs) != 1:
            raise VerifyException(f"Expected a single attribute, got {len(attrs)}")
        self.constr.verify(attrs[0], constraint_context)

    def get_resolved_variables(self) -> set[str]:
        return self.constr.get_resolved_variables()

    def can_infer(self, constraint_names: set[str]) -> bool:
        return self.constr.can_infer(constraint_names)

    def infer(
        self, length: int, constraint_context: ConstraintContext
    ) -> list[Attribute]:
        return [self.constr.infer(constraint_context)]


def range_constr_coercion(
    attr: (
        AttributeCovT
        | type[AttributeCovT]
        | GenericAttrConstraint[AttributeCovT]
        | RangeConstraint[AttributeCovT]
    ),
) -> RangeConstraint[AttributeCovT]:
    if isinstance(attr, RangeConstraint):
        return attr
    return RangeOf(attr_constr_coercion(attr))


def single_range_constr_coercion(
    attr: AttributeCovT | type[AttributeCovT] | GenericAttrConstraint[AttributeCovT],
) -> RangeConstraint[AttributeCovT]:
    return SingleOf(attr_constr_coercion(attr))
