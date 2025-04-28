from __future__ import annotations

import abc
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence, Set
from dataclasses import KW_ONLY, dataclass, field
from inspect import isclass
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast

from typing_extensions import assert_never

from xdsl.ir import (
    Attribute,
    AttributeCovT,
    AttributeInvT,
    ParametrizedAttribute,
    TypedAttribute,
)
from xdsl.utils.exceptions import VerifyException
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

    def set_variable(self, key: str, attr: Attribute):
        self._variables[key] = attr

    def set_range_variable(self, key: str, attr: tuple[Attribute, ...]):
        self._range_variables[key] = attr

    def set_int_variable(self, key: str, i: int):
        self._int_variables[key] = i

    @property
    def variables(self) -> Set[str]:
        return self._variables.keys()

    @property
    def range_variables(self) -> Set[str]:
        return self._range_variables.keys()

    @property
    def int_variables(self) -> Set[str]:
        return self._int_variables.keys()

    def copy(self):
        return ConstraintContext(
            self._variables.copy(),
            self._range_variables.copy(),
            self._int_variables.copy(),
        )

    def update(self, other: ConstraintContext):
        self._variables.update(other._variables)
        self._range_variables.update(other._range_variables)
        self._int_variables.update(other._int_variables)


_AttributeCovT = TypeVar("_AttributeCovT", bound=Attribute, covariant=True)

ConstraintVariableType: TypeAlias = Attribute | Sequence[Attribute] | int
"""
Possible types that a constraint variable can have.
"""


_T = TypeVar("_T")


@dataclass(frozen=True)
class VarExtractor(Generic[_T], abc.ABC):
    """
    A VarExtractor describes how to extract the value of a constraint variable
    from a value of type _T. The result of the extraction must be a subtype of
    ConstraintVariableType.
    """

    @abstractmethod
    def extract_var(self, a: _T) -> ConstraintVariableType: ...

    def flatten(self) -> Iterator[VarExtractor[_T]]:
        """Helper function to merge extractors, should only be overloaded by MergeExtractor"""
        yield self


@dataclass(frozen=True)
class MergeExtractor(VarExtractor[_T]):
    variable: str
    extractors: tuple[VarExtractor[_T], ...]
    """Extractors to merge, should not be empty."""

    def extract_var(self, a: _T) -> ConstraintVariableType:
        res = set(x.extract_var(a) for x in self.extractors)
        if len(res) == 1:
            return res.pop()
        else:
            raise ValueError(
                f"Value of variable {self.variable} could not be uniquely extracted.\n"
                f"Possible values are: {{{', '.join(sorted(str(x) for x in res))}}}"
            )

    def flatten(self) -> Iterator[VarExtractor[_T]]:
        return iter(self.extractors)


def merge_extractors(variable: str, *extractors: VarExtractor[_T]) -> VarExtractor[_T]:
    """
    Helper method used for merging dictionaries of extractors.
    Must not be called with no extractors.
    """
    if len(extractors) == 1:
        return extractors[0]
    extractor_tuple = tuple(x for ex in extractors for x in ex.flatten())
    return MergeExtractor(variable, extractor_tuple)


def merge_extractor_dicts(
    *dicts: dict[str, VarExtractor[_T]],
) -> dict[str, VarExtractor[_T]]:
    """
    Merge dictionaries of extractors.
    The keys of the resulting dictionary is equal to the union of the keys
    of the input dictionaries, with duplicated values being merged to a
    single MergeExtractor
    """
    keys = set(key for d in dicts for key in d)
    return {v: merge_extractors(v, *(d[v] for d in dicts if v in d)) for v in keys}


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

    def get_variable_extractors(self) -> dict[str, VarExtractor[AttributeCovT]]:
        """
        Get a dictionary of constraint variables to extractors for these variables,
        which provide a method to obtain the value of each constraint variable from
        the value of the attribute which is verified by this constraint.
        """
        return {}

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
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
        raise ValueError("Cannot infer attribute from constraint")

    def get_unique_base(self) -> type[Attribute] | None:
        """Get the unique base type that can satisfy the constraint, if any."""
        return None

    def __or__(
        self, value: GenericAttrConstraint[_AttributeCovT], /
    ) -> AnyOf[AttributeCovT | _AttributeCovT]:
        return AnyOf((self, value))

    def __and__(
        self, value: GenericAttrConstraint[AttributeCovT], /
    ) -> AllOf[AttributeCovT]:
        return AllOf((self, value))


AttrConstraint: TypeAlias = GenericAttrConstraint[Attribute]
ConstraintVariableTypeT = TypeVar(
    "ConstraintVariableTypeT", bound=ConstraintVariableType
)


@dataclass(frozen=True)
class IdExtractor(VarExtractor[ConstraintVariableTypeT]):
    """
    An extractor that simply returns its input.
    """

    def extract_var(self, a: ConstraintVariableTypeT) -> ConstraintVariableType:
        return a


TypedAttributeCovT = TypeVar("TypedAttributeCovT", bound=TypedAttribute, covariant=True)
TypedAttributeT = TypeVar("TypedAttributeT", bound=TypedAttribute)


@dataclass(frozen=True)
class TypedAttributeConstraint(GenericAttrConstraint[TypedAttributeCovT]):
    """
    Constrains the type of a typed attribute.
    """

    attr_constraint: GenericAttrConstraint[TypedAttributeCovT]

    type_constraint: GenericAttrConstraint[Attribute]

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        self.attr_constraint.verify(attr, constraint_context)
        if not isinstance(attr, TypedAttribute):
            raise VerifyException(f"attribute {attr} expected to be a TypedAttribute")
        self.type_constraint.verify(attr.get_type(), constraint_context)

    @dataclass(frozen=True)
    class _Extractor(VarExtractor[TypedAttributeT]):
        inner: VarExtractor[Attribute]

        def extract_var(self, a: TypedAttributeT) -> ConstraintVariableType:
            return self.inner.extract_var(a.get_type())

    def get_variable_extractors(self) -> dict[str, VarExtractor[TypedAttributeCovT]]:
        return merge_extractor_dicts(
            self.attr_constraint.get_variable_extractors(),
            {
                v: self._Extractor(r)
                for v, r in self.type_constraint.get_variable_extractors().items()
            },
        )

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.attr_constraint.can_infer(var_constraint_names)

    def infer(self, context: ConstraintContext) -> TypedAttributeCovT:
        return self.attr_constraint.infer(context)


@dataclass(frozen=True)
class VarConstraint(GenericAttrConstraint[AttributeCovT]):
    """
    Constrain an attribute with the given constraint, and constrain all occurences
    of this constraint (i.e, sharing the same name) to be equal.
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
        ctx_attr = constraint_context.get_variable(self.name)
        if ctx_attr is not None:
            if attr != ctx_attr:
                raise VerifyException(
                    f"attribute {constraint_context.get_variable(self.name)} expected from variable "
                    f"'{self.name}', but got {attr}"
                )
        else:
            self.constraint.verify(attr, constraint_context)
            constraint_context.set_variable(self.name, attr)

    def get_variable_extractors(self) -> dict[str, VarExtractor[AttributeCovT]]:
        return merge_extractor_dicts(
            {self.name: IdExtractor()}, self.constraint.get_variable_extractors()
        )

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        v = context.get_variable(self.name)
        return cast(AttributeCovT, v)

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.name in var_constraint_names

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

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return True

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        return self.attr

    def get_unique_base(self) -> type[Attribute] | None:
        return type(self.attr)


@dataclass(frozen=True)
class BaseAttr(Generic[AttributeCovT], GenericAttrConstraint[AttributeCovT]):
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

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return (
            is_runtime_final(self.attr)
            and issubclass(self.attr, ParametrizedAttribute)
            and not self.attr.get_irdl_definition().parameters
        )

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        assert issubclass(self.attr, ParametrizedAttribute)
        attr = self.attr.new(())
        return attr

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
    assert_never(attr)


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

    attr_constrs: tuple[GenericAttrConstraint[AttributeCovT], ...]
    """The list of constraints that are checked."""

    def __init__(
        self,
        attr_constrs: Sequence[
            AttributeCovT | type[AttributeCovT] | GenericAttrConstraint[AttributeCovT]
        ],
    ):
        constrs: tuple[GenericAttrConstraint[AttributeCovT], ...] = tuple(
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

    def get_variable_extractors(self) -> dict[str, VarExtractor[AttributeCovT]]:
        if len(self.attr_constrs) == 1:
            return self.attr_constrs[0].get_variable_extractors()
        return dict()

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

    def get_variable_extractors(self) -> dict[str, VarExtractor[AttributeCovT]]:
        return merge_extractor_dicts(
            *(constr.get_variable_extractors() for constr in self.attr_constrs)
        )

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return any(
            constr.can_infer(var_constraint_names) for constr in self.attr_constrs
        )

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        for constr in self.attr_constrs:
            if constr.can_infer(context.variables):
                return constr.infer(context)
        raise ValueError("Cannot infer attribute from constraint")

    def get_unique_base(self) -> type[Attribute] | None:
        # This could be improved if we keep track of all the possible base types for
        # each constraint.
        for constr in self.attr_constrs:
            base = constr.get_unique_base()
            if base is not None:
                return base
        return None

    def __and__(
        self, value: GenericAttrConstraint[AttributeCovT], /
    ) -> AllOf[AttributeCovT]:
        return AllOf((*self.attr_constrs, value))


ParametrizedAttributeT = TypeVar("ParametrizedAttributeT", bound=ParametrizedAttribute)
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
        if len(self.param_constrs) != len(attr.parameters):
            raise VerifyException(
                f"{len(self.param_constrs)} parameters expected, "
                f"but got {len(attr.parameters)}"
            )
        for idx, param_constr in enumerate(self.param_constrs):
            param_constr.verify(attr.parameters[idx], constraint_context)

    @dataclass(frozen=True)
    class _Extractor(VarExtractor[ParametrizedAttributeT]):
        idx: int
        inner: VarExtractor[Attribute]

        def extract_var(self, a: ParametrizedAttributeT) -> ConstraintVariableType:
            return self.inner.extract_var(a.parameters[self.idx])

    def get_variable_extractors(
        self,
    ) -> dict[str, VarExtractor[ParametrizedAttributeCovT]]:
        dicts: Generator[dict[str, VarExtractor[ParametrizedAttributeCovT]]] = (
            {
                v: self._Extractor(i, r)
                for v, r in param_constr.get_variable_extractors().items()
            }
            for i, param_constr in enumerate(self.param_constrs)
        )
        return merge_extractor_dicts(*dicts)

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return is_runtime_final(self.base_attr) and all(
            constr.can_infer(var_constraint_names) for constr in self.param_constrs
        )

    def infer(self, context: ConstraintContext) -> ParametrizedAttributeCovT:
        params = tuple(constr.infer(context) for constr in self.param_constrs)
        attr = self.base_attr.new(params)
        return attr

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

    def get_variable_extractors(self) -> dict[str, VarExtractor[AttributeCovT]]:
        return self.constr.get_variable_extractors()

    def get_unique_base(self) -> type[Attribute] | None:
        return self.constr.get_unique_base()

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.constr.can_infer(var_constraint_names)

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        return self.constr.infer(context)


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

    def get_length_extractors(
        self,
    ) -> dict[str, VarExtractor[int]]:
        """
        Get a dictionary of variables that can be solved from this attribute.
        """
        return dict()

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
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
        raise ValueError("Cannot infer attribute from constraint")


class AnyInt(IntConstraint):
    """
    Constraint that is verified by all integers.
    """

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        pass


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

    def get_length_extractors(
        self,
    ) -> dict[str, VarExtractor[int]]:
        return {self.name: IdExtractor()}

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.name in var_constraint_names

    def infer(
        self,
        context: ConstraintContext,
    ) -> int:
        v = context.get_int_variable(self.name)
        assert isinstance(v, int)
        return v


@dataclass(frozen=True)
class GenericRangeConstraint(Generic[AttributeCovT], ABC):
    """Constrain a range of attributes to certain values."""

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

    def get_variable_extractors(
        self,
    ) -> dict[str, VarExtractor[Sequence[AttributeCovT]]]:
        """
        Get a dictionary of constraint variables to extractors for these variables,
        which provide a method to obtain the value of each constraint variable from
        the value of the attribute range which is verified by this constraint.
        """
        return {}

    def get_length_extractors(
        self,
    ) -> dict[str, VarExtractor[int]]:
        """
        Get a dictionary of variables that can be solved using the length of the range.
        """
        return dict()

    def can_infer(self, var_constraint_names: Set[str], *, length_known: bool) -> bool:
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
        raise ValueError("Cannot infer attribute from constraint")


RangeConstraint: TypeAlias = GenericRangeConstraint[Attribute]


@dataclass(frozen=True)
class RangeVarConstraint(GenericRangeConstraint[AttributeCovT]):
    """
    Constrain an attribute range with the given constraint, and constrain all occurences
    of this constraint (i.e, sharing the same name) to be equal.
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""

    constraint: GenericRangeConstraint[AttributeCovT]
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

    def get_variable_extractors(
        self,
    ) -> dict[str, VarExtractor[Sequence[AttributeCovT]]]:
        return {self.name: IdExtractor[Sequence[AttributeCovT]]()}

    def can_infer(self, var_constraint_names: Set[str], *, length_known: bool) -> bool:
        return self.name in var_constraint_names

    def infer(
        self, context: ConstraintContext, *, length: int | None
    ) -> Sequence[AttributeCovT]:
        v = context.get_range_variable(self.name)
        return cast(Sequence[AttributeCovT], v)


@dataclass(frozen=True)
class RangeOf(GenericRangeConstraint[AttributeCovT]):
    """
    Constrain each element in a range to satisfy a given constraint.
    """

    constr: GenericAttrConstraint[AttributeCovT]
    _: KW_ONLY
    length: IntConstraint = field(default_factory=AnyInt)

    def verify(
        self,
        attrs: Sequence[Attribute],
        constraint_context: ConstraintContext,
    ) -> None:
        for a in attrs:
            self.constr.verify(a, constraint_context)
        self.length.verify(len(attrs), constraint_context)

    def get_length_extractors(self) -> dict[str, VarExtractor[int]]:
        return self.length.get_length_extractors()

    def can_infer(self, var_constraint_names: Set[str], *, length_known: bool) -> bool:
        return (
            length_known or self.length.can_infer(var_constraint_names)
        ) and self.constr.can_infer(var_constraint_names)

    def infer(
        self,
        context: ConstraintContext,
        *,
        length: int | None,
    ) -> Sequence[AttributeCovT]:
        if length is None:
            length = self.length.infer(context)
        attr = self.constr.infer(context)
        return (attr,) * length


@dataclass(frozen=True)
class SingleOf(GenericRangeConstraint[AttributeCovT]):
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

    @dataclass(frozen=True)
    class _Extractor(VarExtractor[Sequence[AttributeInvT]]):
        inner: VarExtractor[AttributeInvT]

        def extract_var(self, a: Sequence[AttributeInvT]) -> ConstraintVariableType:
            return self.inner.extract_var(a[0])

    def get_variable_extractors(
        self,
    ) -> dict[str, VarExtractor[Sequence[AttributeCovT]]]:
        return {
            v: self._Extractor(r)
            for v, r in self.constr.get_variable_extractors().items()
        }

    def can_infer(
        self, var_constraint_names: Set[str], *, length_known: int | None
    ) -> bool:
        return self.constr.can_infer(var_constraint_names)

    def infer(
        self, context: ConstraintContext, *, length: int | None
    ) -> Sequence[AttributeCovT]:
        return (self.constr.infer(context),)


def range_constr_coercion(
    attr: (
        AttributeCovT
        | type[AttributeCovT]
        | GenericAttrConstraint[AttributeCovT]
        | GenericRangeConstraint[AttributeCovT]
    ),
) -> GenericRangeConstraint[AttributeCovT]:
    if isinstance(attr, GenericRangeConstraint):
        return attr
    return RangeOf(attr_constr_coercion(attr), length=AnyInt())


def single_range_constr_coercion(
    attr: AttributeCovT | type[AttributeCovT] | GenericAttrConstraint[AttributeCovT],
) -> GenericRangeConstraint[AttributeCovT]:
    return SingleOf(attr_constr_coercion(attr))
