from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Reversible,
    Sequence,
)
from dataclasses import dataclass, field
from io import StringIO
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    NoReturn,
    TypeAlias,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import Self, TypeVar

from xdsl.dialect_interfaces import DialectInterface
from xdsl.traits import IsTerminator, NoTerminator, OpTrait, OpTraitInvT
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from typing_extensions import TypeForm

    from xdsl.irdl import ParamAttrDef
    from xdsl.parser import AttrParser, Parser
    from xdsl.printer import Printer

OpT = TypeVar("OpT", bound="Operation")
DialectInterfaceT = TypeVar("DialectInterfaceT", bound=DialectInterface)


@dataclass
class Dialect:
    """Contains the operations and attributes of a specific dialect"""

    _name: str

    _operations: list[type[Operation]] = field(
        default_factory=list[type["Operation"]], init=True, repr=True
    )
    _attributes: list[type[Attribute]] = field(
        default_factory=list[type["Attribute"]], init=True, repr=True
    )
    _interfaces: list[DialectInterface] = field(
        default_factory=list[DialectInterface], init=True, repr=True
    )

    @property
    def operations(self) -> Iterator[type[Operation]]:
        return iter(self._operations)

    @property
    def attributes(self) -> Iterator[type[Attribute]]:
        return iter(self._attributes)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def split_name(name: str) -> tuple[str, str]:
        try:
            names = name.split(".", 1)
            first, second = names
            return (first, second)
        except ValueError as e:
            raise ValueError(f"Invalid operation or attribute name {name}.") from e

    def get_interface(
        self, interface: type[DialectInterfaceT]
    ) -> DialectInterfaceT | None:
        """
        Return a class that implements the 'interface' if it exists.
        """
        for i in self._interfaces:
            if isinstance(i, interface):
                return i
        return None

    def has_interface(self, interface: type[DialectInterfaceT]) -> bool:
        return self.get_interface(interface) is not None


A = TypeVar("A", bound="Attribute")


@dataclass(frozen=True)
class Attribute(ABC):
    """
    A compile-time value.
    Attributes are used to represent SSA variable types, and can be attached
    on operations to give extra information.
    """

    name: ClassVar[str] = field(init=False, repr=False)
    """The attribute name should be a static field in the attribute classes."""

    def __post_init__(self):
        self._verify()
        if not isinstance(self, Data | ParametrizedAttribute):
            raise TypeError("Attributes should only be Data or ParameterizedAttribute")

    def _verify(self):
        self.verify()

    def verify(self) -> None:
        """
        Check that the attribute parameters satisfy the expected invariants.
        Raise a VerifyException otherwise.
        """
        pass

    def __str__(self) -> str:
        from xdsl.printer import Printer

        res = StringIO()
        printer = Printer(stream=res)
        printer.print_attribute(self)
        return res.getvalue()


@dataclass(frozen=True, init=False)
class BuiltinAttribute(Attribute, ABC):
    """
    This class is used to mark builtin attributes.
    Unlike other attributes in MLIR, parsing of *Builtin* attributes
    is handled directly by the parser.
    Printing of these attributes is handled by the `print_builtin` function, which must
    be implemented by all *Builtin* attributes.
    Attributes outside of the `builtin` dialect should not inherit from `BuiltinAttribute`.
    """

    @abstractmethod
    def print_builtin(self, printer: Printer) -> None:
        """
        Prints the attribute using the supplied printer.
        `BuiltinAttribute`s need not follow the same rules as other attributes, for example
        they do not need to be prefixed by `!` or `#` and do not need to print their name.
        """
        ...


class TypeAttribute(Attribute):
    """
    This class should only be inherited by classes inheriting Attribute.
    This class is only used for printing attributes in the MLIR format,
    inheriting this class prefix the attribute by `!` instead of `#`.
    """

    pass


class OpaqueSyntaxAttribute(Attribute):
    """
    This class should only be inherited by classes inheriting Attribute.
    This class is only used for printing attributes in the opaque form.

    See external [documentation](https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values.).
    """

    pass


class SpacedOpaqueSyntaxAttribute(OpaqueSyntaxAttribute):
    """
    This class should only be inherited by classes inheriting Attribute.
    This class is only used for printing attributes in the opaque form.

    See external [documentation](https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values.).
    """

    pass


DataElement = TypeVar("DataElement", covariant=True, bound=Hashable)

AttributeCovT = TypeVar(
    "AttributeCovT", bound=Attribute, covariant=True, default=Attribute
)
AttributeInvT = TypeVar("AttributeInvT", bound=Attribute, default=Attribute)


@dataclass(frozen=True)
class Data(Attribute, ABC, Generic[DataElement]):
    """An attribute represented by a Python structure."""

    data: DataElement

    @classmethod
    def new(cls: type[Self], params: Any) -> Self:
        """
        Create a new `Data` given its parameter.

        The `params` argument should be of the same type as the `Data` generic
        argument.

        This function should be preferred over `__init__` when instantiating
        attributes in a generic way (i.e., without knowing their concrete type
        statically).
        """
        # Create the new attribute object, without calling its __init__.
        # We do this to allow users to redefine their own __init__.
        attr = cls.__new__(cls)

        # Call the __init__ of Data, which will set the parameters field.
        Data.__init__(attr, params)  # pyright: ignore[reportUnknownMemberType]
        return attr

    @classmethod
    def get(cls, attr: DataElement | Self) -> Self:
        """
        Creates an element of this class from `DataElement`,
        or returns the input when it is already an instance of this class.

        This function is useful for `__init__` methods, for example when a we
        would like to accept either a `StringAttr` or a `str`.
        """
        if not isinstance(attr, cls):
            attr = cls.new(attr)
        return attr

    @classmethod
    @abstractmethod
    def parse_parameter(cls, parser: AttrParser) -> DataElement:
        """Parse the attribute parameter."""

    @abstractmethod
    def print_parameter(self, printer: Printer) -> None:
        """Print the attribute parameter."""


EnumType = TypeVar("EnumType", bound=StrEnum)


def _check_enum_constraints(
    enum_class: type[EnumAttribute[EnumType] | BitEnumAttribute[EnumType]],
) -> None:
    """
    This hook first checks two constraints, enforced to keep the implementation
    reasonable, until more complex use cases appear. It then stores the Enum type
    used by the subclass to use in parsing/printing.

    The constraints are:

    - Only direct, specialized inheritance is allowed. That is, using a subclass
    of EnumAttribute as a base class is *not supported*.
      This simplifies type-hacking code and I don't see it being too restrictive
      anytime soon.
    """
    orig_bases = getattr(enum_class, "__orig_bases__")
    enumattr = next(
        b
        for b in orig_bases
        if get_origin(b) is EnumAttribute or get_origin(b) is BitEnumAttribute
    )
    enum_type = get_args(enumattr)[0]
    if isinstance(enum_type, TypeVar):
        raise TypeError("Only direct inheritance from EnumAttribute is allowed.")

    enum_class.enum_type = enum_type


class EnumAttribute(Data[EnumType]):
    """
    Core helper for Enum Attributes. Takes a StrEnum type parameter, and defines
    parsing/printing automatically from its values, restricted to be parsable as
    identifiers.

    example:
    ```python
    class MyEnum(StrEnum):
        First = auto()
        Second = auto()

    class MyEnumAttribute(EnumAttribute[MyEnum], SpacedOpaqueSyntaxAttribute):
        name = "example.my_enum"
    ```
    To use this attribute suffices to have a textual representation
    of `example<my_enum first>` and ``example<my_enum second>``

    """

    enum_type: ClassVar[type[StrEnum]]

    def __init_subclass__(cls) -> None:
        _check_enum_constraints(cls)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_identifier_or_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> EnumType:
        return cast(EnumType, parser.parse_str_enum(cls.enum_type))


@dataclass(frozen=True, init=False)
class BitEnumAttribute(Data[tuple[EnumType, ...]], Generic[EnumType]):
    """
    Core helper for BitEnumAttributes. Takes a StrEnum type parameter, and
    defines parsing/printing automatically from its values.

    Additionally, two values can be given to designate all/none bits being set.

    example:
    ```python
    class MyBitEnum(StrEnum):
        First = auto()
        Second = auto()

    class MyBitEnumAttribute(BitEnumAttribute[MyBitEnum]):
        name = "example.my_bit_enum"
        none_value = "none"
        all_value = "all"

    """

    enum_type: ClassVar[type[StrEnum]]
    none_value: ClassVar[str | None] = None
    all_value: ClassVar[str | None] = None

    def __init__(self, flags: None | Sequence[EnumType] | str) -> None:
        flags_: set[EnumType]
        match flags:
            case self.none_value | None:
                flags_ = set()
            case self.all_value:
                flags_ = cast(set[EnumType], set(self.enum_type))
            case other if isinstance(other, str):
                raise TypeError(
                    f"expected string parameter to be one of {self.none_value} or {self.all_value}, got {other}"
                )
            case other:
                assert not isinstance(other, str)
                flags_ = set(other)

        super().__init__(tuple(flags_))

    def __init_subclass__(cls) -> None:
        _check_enum_constraints(cls)

    @property
    def flags(self) -> set[EnumType]:
        return set(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[EnumType, ...]:
        def parse_optional_element() -> set[EnumType] | None:
            if (
                cls.none_value is not None
                and parser.parse_optional_keyword(cls.none_value) is not None
            ):
                return set()
            if (
                cls.all_value is not None
                and parser.parse_optional_keyword(cls.all_value) is not None
            ):
                return set(cast(Iterable[EnumType], cls.enum_type))
            value = parser.parse_optional_str_enum(cls.enum_type)
            if value is None:
                return None

            return {cast(type[EnumType], cls.enum_type)(value)}

        def parse_element() -> set[EnumType]:
            if (
                cls.none_value is not None
                and parser.parse_optional_keyword(cls.none_value) is not None
            ):
                return set()
            if (
                cls.all_value is not None
                and parser.parse_optional_keyword(cls.all_value) is not None
            ):
                return set(cast(Iterable[EnumType], cls.enum_type))
            value = parser.parse_str_enum(cls.enum_type)
            return {cast(type[EnumType], cls.enum_type)(value)}

        with parser.in_angle_brackets():
            flags: list[set[EnumType]] | None = (
                parser.parse_optional_undelimited_comma_separated_list(
                    parse_optional_element, parse_element
                )
            )
            if flags is None:
                return tuple()

            res = set[EnumType]()

            for flag_set in flags:
                res |= flag_set

            return tuple(res)

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            flags = self.data
            if len(flags) == 0 and self.none_value is not None:
                printer.print_string(self.none_value)
            elif len(flags) == len(self.enum_type) and self.all_value is not None:
                printer.print_string(self.all_value)
            else:
                # make sure we emit flags in a consistent order
                printer.print_list(
                    tuple(flag.value for flag in self.enum_type if flag in flags),
                    printer.print_string,
                    ",",
                )


@dataclass(frozen=True, init=False)
class ParametrizedAttribute(Attribute):
    """An attribute parametrized by other attributes."""

    def __init__(self, *parameters: Any):
        irdl_def = self.get_irdl_definition()
        for (f, d), param in zip(irdl_def.parameters, parameters, strict=True):
            if d.converter is not None:
                param = d.converter(param)
            object.__setattr__(self, f, param)
        super().__init__()

    @property
    def parameters(self) -> tuple[Attribute, ...]:
        return (
            *(
                self.__getattribute__(field)
                for field, _ in self.get_irdl_definition().parameters
            ),
        )

    @classmethod
    def new(cls: type[Self], params: Sequence[Attribute]) -> Self:
        """
        Create a new `ParametrizedAttribute` given its parameters.

        This function should be preferred over `__init__` when instantiating
        attributes in a generic way (i.e., without knowing their concrete type
        statically).
        """
        # Create the new attribute object, without calling its __init__.
        # We do this to allow users to redefine their own __init__.
        attr = cls.__new__(cls)

        # Set the parameters based on the definition
        for (f, _), param in zip(
            cls.get_irdl_definition().parameters, params, strict=True
        ):
            object.__setattr__(attr, f, param)
        attr.__post_init__()

        return attr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        return parser.parse_paramattr_parameters()

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print_paramattr_parameters(self.parameters)

    @classmethod
    def get_irdl_definition(cls) -> ParamAttrDef:
        """Get the IRDL attribute definition."""
        ...

    def _verify(self):
        # Verifier generated by irdl_attr_def
        t: type[ParametrizedAttribute] = type(self)
        attr_def = t.get_irdl_definition()
        attr_def.verify(self)
        super()._verify()


class TypedAttribute(ParametrizedAttribute, ABC):
    """
    An attribute with a type.
    """

    @classmethod
    def get_type_index(cls) -> int: ...

    def get_type(self) -> Attribute:
        return self.parameters[self.get_type_index()]

    @staticmethod
    def parse_with_type(
        parser: AttrParser,
        type: Attribute,
    ) -> TypedAttribute:
        """
        Parse the attribute with the given type.
        """
        ...

    @abstractmethod
    def print_without_type(self, printer: Printer): ...


@dataclass(repr=False)
class Use:
    """The use of a SSA value."""

    _operation: Operation
    """The operation using the value."""

    _index: int
    """The index of the operand using the value in the operation."""

    _prev_use: Use | None = None
    """The previous use of the value in the use list."""

    _next_use: Use | None = None
    """The next use of the value in the use list."""

    @property
    def operation(self) -> Operation:
        """The operation using the value."""
        return self._operation

    @property
    def index(self) -> int:
        """The index of the operand using the value in the operation."""
        return self._index

    def __repr__(self) -> str:
        return (
            f"<Use {id(self)}(_operation={_short_repr(self.operation)}, "
            f"_index={self.index}, _prev_use={_short_repr(self._prev_use)}, "
            f"_next_use={_short_repr(self._next_use)})>"
        )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class IRUses(Iterable[Use]):
    """
    Multi-pass iterable of the uses of an IR value (SSAValue or Block).
    """

    ir: IRWithUses

    def __iter__(self):
        use = self.ir.first_use
        while use is not None:
            yield use
            use = use._next_use  # pyright: ignore[reportPrivateUsage]

    def __bool__(self) -> bool:
        """Returns `True` if there are operations in this block."""
        return self.ir.first_use is not None

    def get_length(self) -> int:
        """
        Returns the number of uses.
        We do not expose it as `__len__` as it is expensive to compute `O(len)`.
        """
        use = self.ir.first_use
        count = 0
        while use is not None:
            count += 1
            use = use._next_use  # pyright: ignore[reportPrivateUsage]
        return count


@dataclass(eq=False)
class IRWithUses(ABC):
    """IRNode which stores a list of its uses."""

    first_use: Use | None = field(init=False, default=None, repr=False)
    """The first use of the value in the use list."""

    @property
    def uses(self) -> IRUses:
        """Returns an iterable of all uses of the value."""
        return IRUses(self)

    def add_use(self, use: Use):
        """Add a new use of the value."""
        first_use = self.first_use
        use._next_use = first_use  # pyright: ignore[reportPrivateUsage]
        use._prev_use = None  # pyright: ignore[reportPrivateUsage]
        if first_use is not None:
            first_use._prev_use = use  # pyright: ignore[reportPrivateUsage]
        self.first_use = use

    def remove_use(self, use: Use):
        """Remove a use of the value."""
        prev_use = use._prev_use  # pyright: ignore[reportPrivateUsage]
        next_use = use._next_use  # pyright: ignore[reportPrivateUsage]
        if prev_use is not None:
            prev_use._next_use = next_use  # pyright: ignore[reportPrivateUsage]
        if next_use is not None:
            next_use._prev_use = prev_use  # pyright: ignore[reportPrivateUsage]

        if prev_use is None:
            self.first_use = next_use

    def has_one_use(self) -> bool:
        """Returns true if the value has exactly one use."""
        first_use = self.first_use
        return (
            first_use is not None and first_use._next_use is None  # pyright: ignore[reportPrivateUsage]
        )

    def has_more_than_one_use(self) -> bool:
        """Returns true if the value has more than one use."""
        if (first_use := self.first_use) is None:
            return False
        return first_use._next_use is not None  # pyright: ignore[reportPrivateUsage]

    def get_unique_use(self) -> Use | None:
        """
        Returns the single use of the value, or None if there are no uses or
        more than one use.
        """
        if (first_use := self.first_use) is None:
            return None
        if first_use._next_use is not None:  # pyright: ignore[reportPrivateUsage]
            return None
        return first_use

    def get_user_of_unique_use(self) -> Operation | None:
        """
        Returns the user of the single use of the value.
        If there are no uses or more than one use, returns None.
        """
        if (use := self.get_unique_use()) is not None:
            return use.operation
        return None


_VALUE_NAME_PATTERN = re.compile(r"([A-Za-z_$.-][\w$.-]*)")
"""Pattern to check if a name is valid for an SSAValue or Block."""

_VALUE_NAME_SUFFIX_PATTERN = re.compile(r"(_\d+)$")
"""This pattern is used to remove the suffix from an SSAValue or Block name."""


@dataclass(eq=False)
class IRWithName(ABC):
    _name: str | None = field(init=False, default=None)

    @property
    def name_hint(self) -> str | None:
        return self._name

    @name_hint.setter
    def name_hint(self, name: str | None):
        self._name = self.extract_valid_name(name)

    @classmethod
    def is_valid_name(cls, name: str | None):
        """
        Returns `True` if `name` is None or a valid name for an identifier.
        """
        return name is None or _VALUE_NAME_PATTERN.fullmatch(name)

    @overload
    @classmethod
    def extract_valid_name(cls, name: str) -> str: ...

    @overload
    @classmethod
    def extract_valid_name(cls, name: None) -> None: ...

    @classmethod
    def extract_valid_name(cls, name: str | None) -> str | None:
        """
        If the name is valid, extracts the name before an optional `_\\d+` suffix.
        Raises ValueError otherwise.
        """
        if name is None:
            return
        if _VALUE_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"Invalid {cls.__name__} name format `{name}`.",
                r"Make sure names contain only characters of [A-Za-z0-9_$.-] and don't start with a number.",
            )

        if match := _VALUE_NAME_SUFFIX_PATTERN.search(name):
            # Remove `_` followed by numbers at the end of the name
            return name[: match.start()]

        return name


@dataclass(eq=False)
class SSAValue(IRWithUses, IRWithName, ABC, Generic[AttributeCovT]):
    """
    A reference to an SSA variable.
    An SSA variable is either an operation result, or a basic block argument.
    """

    _type: AttributeCovT
    """Each SSA variable is associated to a type."""

    @property
    def type(self) -> AttributeCovT:
        return self._type

    @property
    @abstractmethod
    def owner(self) -> Operation | Block:
        """
        An SSA variable is either an operation result, or a basic block argument.
        This property returns the Operation or Block that currently defines a specific value.
        """
        pass

    @staticmethod
    def get(
        arg: SSAValue | Operation, *, type: TypeForm[AttributeInvT] = Attribute
    ) -> SSAValue[AttributeInvT]:
        """
        Get a new SSAValue from either a SSAValue, or an operation with a single result.
        Checks that the resulting SSAValue is of the supplied type, if provided.
        """
        from xdsl.utils.hints import isa

        match arg:
            case SSAValue():
                if type is Attribute or isa(arg.type, type):
                    return cast(SSAValue[AttributeInvT], arg)
                raise ValueError(
                    f"SSAValue.get: Expected {type} but got SSAValue with type {arg.type}."
                )
            case Operation():
                if len(arg.results) == 1:
                    return SSAValue.get(arg.results[0], type=type)
                raise ValueError(
                    "SSAValue.get: expected operation with a single result."
                )

    def replace_by(self, value: SSAValue) -> None:
        """Replace the value by another value in all its uses."""
        for use in tuple(self.uses):
            use.operation.operands[use.index] = value
        # carry over name if possible
        if value.name_hint is None:
            value.name_hint = self.name_hint
        assert self.first_use is None, "unexpected error in xdsl"

    def replace_by_if(self, value: SSAValue, test: Callable[[Use], bool]):
        """
        Replace the value by another value in all its uses that pass the given test
        function.
        """
        for use in tuple(self.uses):
            if test(use):
                use.operation.operands[use.index] = value
        # carry over name if possible
        if value.name_hint is None:
            value.name_hint = self.name_hint

    def erase(self, safe_erase: bool = True) -> None:
        """
        Erase the value.
        If safe_erase is True, then check that no operations use the value anymore.
        If safe_erase is False, then replace its uses by an ErasedSSAValue.
        """
        if safe_erase and self.first_use is not None:
            raise ValueError(
                "Attempting to delete SSA value that still has uses of result "
                f"of operation:\n{self.owner}"
            )
        self.replace_by(ErasedSSAValue(self.type, self))

    def __hash__(self):
        """
        Make SSAValue hashable. Two SSA Values are never the same, therefore
        the use of `id` is allowed here.
        """
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass(eq=False)
class OpResult(SSAValue[AttributeCovT], Generic[AttributeCovT]):
    """A reference to an SSA variable defined by an operation result."""

    op: Operation
    """The operation defining the variable."""

    index: int
    """The index of the result in the defining operation."""

    @property
    def owner(self) -> Operation:
        return self.op

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}[{self.type}]"
            f" name_hint: {self.name_hint},"
            f" index: {self.index},"
            f" operation: {self.op.name},"
            f" uses: {self.uses.get_length()}>"
        )


@dataclass(eq=False)
class BlockArgument(SSAValue[AttributeCovT], Generic[AttributeCovT]):
    """A reference to an SSA variable defined by a basic block argument."""

    block: Block
    """The block defining the variable."""

    index: int
    """The index of the variable in the block arguments."""

    @property
    def owner(self) -> Block:
        return self.block

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}[{self.type}]"
            f" name_hint: {self.name_hint},"
            f" index: {self.index},"
            f" uses: {self.uses.get_length()}>"
        )


@dataclass(eq=False)
class ErasedSSAValue(SSAValue):
    """
    An erased SSA variable.
    This is used during transformations when a SSA variable is destroyed but still used.
    """

    old_value: SSAValue

    @property
    def owner(self) -> Operation | Block:
        return self.old_value.owner


@dataclass(init=False)
class _IRNode(ABC):
    def is_ancestor(self, op: IRNode) -> bool:
        "Returns true if `self` is an ancestor of `op`."
        curr = op
        while curr is not None:
            if curr is self:
                return True
            curr = curr.parent_node
        return False

    def get_toplevel_object(self) -> IRNode:
        """
        Get the ancestor of `self` that has no parent.
        This can be an Operation, Block, or Region.
        """
        current = self
        while (parent := current.parent_node) is not None:
            current = parent
        return current  # pyright: ignore[reportReturnType]

    def is_structurally_equivalent(
        self,
        other: IRNode,
        context: dict[IRNode | SSAValue, IRNode | SSAValue] | None = None,
    ) -> bool:
        """Check if two IR nodes are structurally equivalent."""
        ...

    @property
    @abstractmethod
    def parent_node(self) -> IRNode | None: ...

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


SSAValueCovT = TypeVar(
    "SSAValueCovT", bound=SSAValue[Attribute], default=SSAValue[Attribute]
)


class SSAValues(tuple[SSAValueCovT, ...], Generic[SSAValueCovT]):
    """
    A helper data structure for a sequence of SSAValues.
    """

    @property
    def types(self):
        return tuple(o.type for o in self)

    @overload
    def __getitem__(self, idx: int) -> SSAValueCovT: ...

    @overload
    def __getitem__(self, idx: slice) -> SSAValues[SSAValueCovT]: ...

    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, idx: int | slice
    ) -> SSAValueCovT | SSAValues[SSAValueCovT]:
        if isinstance(idx, int):
            return super().__getitem__(idx)
        else:
            return SSAValues(super().__getitem__(idx))


@dataclass
class OpOperands(Sequence[SSAValue]):
    """
    A view of the operand list of an operation.
    Any modification to the view is reflected on the operation.
    """

    _op: Operation
    """The operation owning the operands."""

    @overload
    def __getitem__(self, idx: int) -> SSAValue: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[SSAValue]: ...

    def __getitem__(self, idx: int | slice) -> SSAValue | Sequence[SSAValue]:
        return self._op._operands[idx]  # pyright: ignore[reportPrivateUsage]

    def __setitem__(self, idx: int, operand: SSAValue) -> None:
        operands = self._op._operands  # pyright: ignore[reportPrivateUsage]
        operand_uses = self._op._operand_uses  # pyright: ignore[reportPrivateUsage]
        operands[idx].remove_use(operand_uses[idx])
        operand.add_use(operand_uses[idx])
        new_operands = SSAValues((*operands[:idx], operand, *operands[idx + 1 :]))
        self._op._operands = new_operands  # pyright: ignore[reportPrivateUsage]

    def __iter__(self) -> Iterator[SSAValue]:
        return iter(self._op._operands)  # pyright: ignore[reportPrivateUsage]

    def __len__(self) -> int:
        return len(self._op._operands)  # pyright: ignore[reportPrivateUsage]

    def __eq__(self, other: object):
        if not isinstance(other, OpOperands):
            return False
        return (
            self._op._operands  # pyright: ignore[reportPrivateUsage]
            == other._op._operands  # pyright: ignore[reportPrivateUsage]
        )

    def __hash__(self):
        return hash(self._op._operands)  # pyright: ignore[reportPrivateUsage]


class OpTraits(Iterable[OpTrait]):
    """
    An operation's traits.
    Some operations have mutually recursive traits, such as one is always the parent
    operation of the other.
    For this case, the operation's traits can be declared lazily and resolved only at
    the first use.
    """

    gen_traits: Callable[[], tuple[OpTrait, ...]]
    """
    Factory method that lazily populates the traits on first use.
    """
    _traits: frozenset[OpTrait] | None
    """
    The traits of this operation, can be updated via `add_trait`.
    """

    def __init__(self, gen_traits: Callable[[], tuple[OpTrait, ...]]) -> None:
        self.gen_traits = gen_traits
        self._traits = None

    @property
    def traits(self) -> frozenset[OpTrait]:
        """Returns a copy of this instance's traits."""
        if self._traits is None:
            self._traits = frozenset(self.gen_traits())
        return self._traits

    def add_trait(self, trait: OpTrait):
        """Adds a trait to the class."""
        self._traits = self.traits.union((trait,))

    def __iter__(self) -> Iterator[OpTrait]:
        return iter(self.traits)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, OpTraits) and self.traits == value.traits


@dataclass(eq=False, unsafe_hash=False, repr=False)
class Operation(_IRNode):
    """A generic operation. Operation definitions inherit this class."""

    name: ClassVar[str]
    """The operation name. Should be a static member of the class"""

    _operands: SSAValues = field(default=SSAValues())
    """The operation operands."""

    _operand_uses: tuple[Use, ...] = field(default=())
    """
    The uses for each operand.
    They are stored separately from the operands to allow for more efficient
    access to the operand SSAValue.
    """

    results: SSAValues[OpResult] = field(default=SSAValues())
    """The results created by the operation."""

    _successors: tuple[Block, ...] = field(default=())
    """
    The basic blocks that the operation may give control to.
    This list should be empty for non-terminator operations.
    """

    _successor_uses: tuple[Use, ...] = field(default=())
    """
    The uses for each successor.
    They are stored separately from the successors to allow for more efficient
    access to the successor Block.
    """

    properties: dict[str, Attribute] = field(default_factory=dict[str, Attribute])
    """
    The properties attached to the operation.
    Properties are inherent to the definition of an operation's semantics, and
    thus cannot be discarded by transformations.
    """

    attributes: dict[str, Attribute] = field(default_factory=dict[str, Attribute])
    """The attributes attached to the operation."""

    regions: tuple[Region, ...] = field(default=())
    """Regions arguments of the operation."""

    parent: Block | None = field(default=None)
    """The block containing this operation."""

    _next_op: Operation | None = field(default=None)
    """Next operation in block containing this operation."""

    _prev_op: Operation | None = field(default=None)
    """Previous operation in block containing this operation."""

    traits: ClassVar[OpTraits]
    """
    Traits attached to an operation definition.
    This is a static field, and is made empty by default by PyRDL if not set
    by the operation definition.
    """

    @property
    def parent_node(self) -> IRNode | None:
        return self.parent

    @property
    def result_types(self) -> Sequence[Attribute]:
        return tuple(r.type for r in self.results)

    @property
    def operand_types(self) -> Sequence[Attribute]:
        return tuple(operand.type for operand in self.operands)

    def parent_op(self) -> Operation | None:
        if p := self.parent_region():
            return p.parent
        return None

    def parent_region(self) -> Region | None:
        if (p := self.parent_block()) is not None:
            return p.parent
        return None

    def parent_block(self) -> Block | None:
        return self.parent

    @property
    def next_op(self) -> Operation | None:
        """
        Next operation in block containing this operation.
        """
        return self._next_op

    def _insert_next_op(self, new_op: Operation) -> None:
        """
        Sets `next_op` on `self`, and `prev_op` on `self.next_op`.
        """

        if self._next_op is not None:
            # update next node
            self._next_op._prev_op = new_op

        # set next and previous on new node
        new_op._prev_op = self
        new_op._next_op = self._next_op

        # update self
        self._next_op = new_op

    @property
    def prev_op(self) -> Operation | None:
        """
        Previous operation in block containing this operation.
        """
        return self._prev_op

    def _insert_prev_op(self, new_op: Operation) -> None:
        """
        Sets `prev_op` on `self`, and `next_op` on `self.prev_op`.
        """

        if self._prev_op is not None:
            # update prev node
            self._prev_op._next_op = new_op

        # set next and previous on new node
        new_op._prev_op = self._prev_op
        new_op._next_op = self

        # update self
        self._prev_op = new_op

    @property
    def operands(self) -> OpOperands:
        return OpOperands(self)

    @operands.setter
    def operands(self, new: Sequence[SSAValue]):
        new = SSAValues(new)
        new_uses = tuple(Use(self, idx) for idx in range(len(new)))
        for operand, use in zip(self._operands, self._operand_uses):
            operand.remove_use(use)
        for operand, use in zip(new, new_uses):
            operand.add_use(use)
        self._operands = new
        self._operand_uses = new_uses

    @property
    def successors(self) -> OpSuccessors:
        return OpSuccessors(self)

    @successors.setter
    def successors(self, new: Sequence[Block]):
        new = tuple(new)
        new_uses = tuple(Use(self, idx) for idx in range(len(new)))
        for successor, use in zip(self._successors, self._successor_uses):
            successor.remove_use(use)
        for successor, use in zip(new, new_uses):
            successor.add_use(use)
        self._successors = new
        self._successor_uses = new_uses

    @property
    def successor_uses(self) -> Sequence[Use]:
        return self._successor_uses

    def __post_init__(self):
        assert self.name != ""
        assert isinstance(self.name, str)

    def __init__(
        self,
        *,
        operands: Sequence[SSAValue] = (),
        result_types: Sequence[Attribute] = (),
        properties: Mapping[str, Attribute] = {},
        attributes: Mapping[str, Attribute] = {},
        successors: Sequence[Block] = (),
        regions: Sequence[Region] = (),
    ) -> None:
        super().__init__()

        # This is assumed to exist by Operation.operand setter.
        self.operands = operands

        self.results = SSAValues(
            OpResult(result_type, self, idx)
            for (idx, result_type) in enumerate(result_types)
        )
        self.properties = dict(properties)
        self.attributes = dict(attributes)
        self.successors = list(successors)
        self.regions = ()
        for region in regions:
            self.add_region(region)

        self.__post_init__()

    @classmethod
    def create(
        cls: type[Self],
        *,
        operands: Sequence[SSAValue] = (),
        result_types: Sequence[Attribute] = (),
        properties: Mapping[str, Attribute] = {},
        attributes: Mapping[str, Attribute] = {},
        successors: Sequence[Block] = (),
        regions: Sequence[Region] = (),
    ) -> Self:
        op = cls.__new__(cls)
        Operation.__init__(
            op,
            operands=operands,
            result_types=result_types,
            properties=properties,
            attributes=attributes,
            successors=successors,
            regions=regions,
        )
        return op

    def add_region(self, region: Region) -> None:
        """Add an unattached region to the operation."""
        if region.parent:
            raise ValueError(
                "Cannot add region that is already attached on an operation."
            )
        self.regions += (region,)
        region.parent = self

    def get_region_index(self, region: Region) -> int:
        """Get the region position in the operation."""
        if region.parent is not self:
            raise ValueError("Region is not attached to the operation.")
        return next(
            idx for idx, curr_region in enumerate(self.regions) if curr_region is region
        )

    def detach_region(self, region: int | Region) -> Region:
        """
        Detach a region from the operation.
        Returns the detached region.
        """
        if isinstance(region, Region):
            region_idx = self.get_region_index(region)
        else:
            region_idx = region
            region = self.regions[region_idx]
        region.parent = None
        self.regions = self.regions[:region_idx] + self.regions[region_idx + 1 :]
        return region

    def drop_all_references(self) -> None:
        """
        Drop all references to other operations.
        This function is called prior to deleting an operation.
        """
        self.parent = None
        for operand, use in zip(self._operands, self._operand_uses):
            operand.remove_use(use)
        for region in self.regions:
            region.drop_all_references()

    def walk(
        self, *, reverse: bool = False, region_first: bool = False
    ) -> Iterator[Operation]:
        """
        Iterate all operations contained in the operation (including this one).
        If region_first is set, then the operation regions are iterated before the
        operation. If reverse is set, then the region, block, and operation lists are
        iterated in reverse order.
        """
        if not region_first:
            yield self
        for region in reversed(self.regions) if reverse else self.regions:
            yield from region.walk(reverse=reverse, region_first=region_first)
        if region_first:
            yield self

    def walk_blocks(self, *, reverse: bool = False) -> Iterator[Block]:
        """
        Iterate over all the blocks nested in the region.
        Iterate in reverse order if reverse is True.
        """
        for region in reversed(self.regions) if reverse else self.regions:
            for block in reversed(region.blocks) if reverse else region.blocks:
                yield from block.walk_blocks(reverse=reverse)

    def get_attr_or_prop(self, name: str) -> Attribute | None:
        """
        Get a named attribute or property.
        It first look into the property dictionary, then into the attribute dictionary.
        """
        if name in self.properties:
            return self.properties[name]
        if name in self.attributes:
            return self.attributes[name]
        return None

    def is_before_in_block(self, other_op: Operation) -> bool:
        """
        Return true if the current operation is located strictly before other_op.
        False otherwise.
        """
        if (
            parent_block := self.parent_block()
        ) is None or other_op.parent_block() is not parent_block:
            return False

        op = self.next_op
        while op is not None:
            if op is other_op:
                return True
            op = op.next_op
        return False

    def verify(self, verify_nested_ops: bool = True) -> None:
        for operand in self.operands:
            if isinstance(operand, ErasedSSAValue):
                raise ValueError("Erased SSA value is used by the operation")

        parent_block = self.parent
        parent_region = None if parent_block is None else parent_block.parent

        if self.successors:
            if parent_block is None or parent_region is None:
                raise VerifyException(
                    f"Operation {self.name} with block successors does not belong to a block or a region"
                )

            if parent_block.last_op is not self:
                raise VerifyException(
                    f"Operation {self.name} with block successors must terminate its parent block"
                )

            for succ in self.successors:
                if succ.parent != parent_block.parent:
                    raise VerifyException(
                        f"Operation {self.name} is branching to a block of a different region"
                    )

        if parent_block is not None and parent_region is not None:
            if parent_block.last_op == self:
                if len(parent_region.blocks) == 1:
                    if (
                        parent_op := parent_region.parent
                    ) is not None and not parent_op.has_trait(NoTerminator):
                        if not self.has_trait(IsTerminator):
                            raise VerifyException(
                                f"Operation {self.name} terminates block in "
                                "single-block region but is not a terminator"
                            )
                elif len(parent_region.blocks) > 1:
                    if not self.has_trait(IsTerminator):
                        raise VerifyException(
                            f"Operation {self.name} terminates block in multi-block "
                            "region but is not a terminator"
                        )

        if verify_nested_ops:
            for region in self.regions:
                region.verify()

        # Custom verifier
        try:
            self.verify_()
        except VerifyException as err:
            self.emit_error(
                f"Operation does not verify: {err}",
                err,
            )

    def verify_(self) -> None:
        pass

    _OperationType = TypeVar("_OperationType", bound="Operation")

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.raise_error(f"Operation {cls.name} does not have a custom format.")

    def print(self, printer: Printer):
        return printer.print_op_with_default_format(self)

    def clone_without_regions(
        self,
        value_mapper: dict[SSAValue, SSAValue] | None = None,
        block_mapper: dict[Block, Block] | None = None,
        *,
        clone_name_hints: bool = True,
        clone_operands: bool = True,
    ) -> Self:
        """Clone an operation, with empty regions instead."""
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}
        if clone_operands:
            operands = tuple(
                value_mapper.get(operand, operand) for operand in self._operands
            )
        else:
            operands = ()
        result_types = self.result_types
        attributes = self.attributes.copy()
        properties = self.properties.copy()
        successors = [
            (block_mapper[successor] if successor in block_mapper else successor)
            for successor in self._successors
        ]
        regions = [Region() for _ in self.regions]
        cloned_op = self.create(
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            properties=properties,
            successors=successors,
            regions=regions,
        )
        for self_result, cloned_result in zip(
            self.results, cloned_op.results, strict=True
        ):
            value_mapper[self_result] = cloned_result
            if clone_name_hints:
                cloned_result.name_hint = self_result.name_hint
        return cloned_op

    def clone(
        self,
        value_mapper: dict[SSAValue, SSAValue] | None = None,
        block_mapper: dict[Block, Block] | None = None,
        *,
        clone_name_hints: bool = True,
        clone_operands: bool = True,
    ) -> Self:
        """Clone an operation with all its regions and operations in them."""
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}
        op = self.clone_without_regions(
            value_mapper,
            block_mapper,
            clone_name_hints=clone_name_hints,
            clone_operands=False,
        )
        for idx, region in enumerate(self.regions):
            region.clone_into(
                op.regions[idx],
                0,
                value_mapper,
                block_mapper,
                clone_name_hints=clone_name_hints,
                clone_operands=False,
            )
        if clone_operands:
            for old, new in zip(self.walk(), op.walk()):
                new.operands = tuple(
                    value_mapper.get(operand, operand) for operand in old.operands
                )
        return op

    @classmethod
    def has_trait(
        cls,
        trait: type[OpTrait] | OpTrait,
        *,
        value_if_unregistered: bool = True,
    ) -> bool:
        """
        Check if the operation implements a trait with the given parameters.
        If the operation is not registered, return value_if_unregisteed instead.
        """
        return cls.get_trait(trait) is not None

    @classmethod
    def get_trait(cls, trait: type[OpTraitInvT] | OpTraitInvT) -> OpTraitInvT | None:
        """
        Return a trait with the given type and parameters, if it exists.
        """
        if isinstance(trait, type):
            for t in cls.traits:
                if isinstance(t, cast(type[OpTraitInvT], trait)):
                    return t
        else:
            for t in cls.traits:
                if t == trait:
                    return cast(OpTraitInvT, t)
        return None

    @classmethod
    def get_traits_of_type(cls, trait_type: type[OpTraitInvT]) -> list[OpTraitInvT]:
        """
        Get all the traits of the given type satisfied by this operation.
        """
        return [t for t in cls.traits if isinstance(t, trait_type)]

    def erase(self, safe_erase: bool = True, drop_references: bool = True) -> None:
        """
        Erase the operation, and remove all its references to other operations.
        If safe_erase is specified, check that the operation results are not used.
        """
        assert self.parent is None, (
            "Operation with parents should first be detached " + "before erasure."
        )
        if drop_references:
            self.drop_all_references()
        for result in self.results:
            result.erase(safe_erase=safe_erase)

    def detach(self):
        """Detach the operation from its parent block."""
        if self.parent is None:
            raise ValueError("Cannot detach a toplevel operation.")
        self.parent.detach_op(self)

    def is_structurally_equivalent(
        self,
        other: IRNode,
        context: dict[IRNode | SSAValue, IRNode | SSAValue] | None = None,
    ) -> bool:
        """
        Check if two operations are structurally equivalent.
        The context is a mapping of IR nodes to IR nodes that are already known
        to be equivalent. This enables checking whether the use dependencies and
        successors are equivalent.
        """
        if context is None:
            context = {}
        if not isinstance(other, Operation):
            return False
        if self.name != other.name:
            return False
        if (
            len(self.operands) != len(other.operands)
            or len(self.results) != len(other.results)
            or len(self.regions) != len(other.regions)
            or len(self.successors) != len(other.successors)
            or self.attributes != other.attributes
            or self.properties != other.properties
        ):
            return False
        if (
            self.parent is not None
            and other.parent is not None
            and context.get(self.parent) != other.parent
        ):
            return False
        if not all(
            context.get(operand, operand) == other_operand
            for operand, other_operand in zip(self.operands, other.operands)
        ):
            return False
        if not all(
            context.get(successor, successor) == other_successor
            for successor, other_successor in zip(self.successors, other.successors)
        ):
            return False
        if not all(
            region.is_structurally_equivalent(other_region, context)
            for region, other_region in zip(self.regions, other.regions)
        ):
            return False
        # Add results of this operation to the context
        for result, other_result in zip(self.results, other.results):
            context[result] = other_result

        return True

    def emit_error(
        self,
        message: str,
        underlying_error: Exception,
    ) -> NoReturn:
        """Emit an error with the given message."""
        from xdsl.utils.diagnostic import Diagnostic

        diagnostic = Diagnostic()
        diagnostic.add_message(self, message)
        diagnostic.raise_exception(self, underlying_error)

    @classmethod
    def dialect_name(cls) -> str:
        return Dialect.split_name(cls.name)[0]

    def __repr__(self) -> str:
        operands = ", ".join(_short_repr(operand) for operand in self.operands)
        results = ", ".join(_short_repr(result) for result in self.results)
        successors = ", ".join(_short_repr(successor) for successor in self.successors)
        regions = ", ".join(_short_repr(region) for region in self.regions)

        return (
            f"<{self.__class__.__name__} {id(self)}("
            f"operands=[{operands}], "
            f"results=[{results}], "
            f"successors=[{successors}], "
            f"properties={repr(self.properties)}, "
            f"attributes={repr(self.attributes)}, "
            f"regions=[{regions}], "
            f"parent={_short_repr(self.parent)}, "
            f"_next_op={_short_repr(self.next_op)}, "
            f"_prev_op={_short_repr(self._prev_op)}"
            ")>"
        )

    def __str__(self) -> str:
        from xdsl.printer import Printer

        res = StringIO()
        printer = Printer(stream=res)
        printer.print_op(self)
        return res.getvalue()

    def __format__(self, format_spec: str, /) -> str:
        desc = str(self)
        if "\n" in desc:
            # Description is multi-line, indent each line
            desc = "\n".join("\t" + line for line in desc.splitlines())
            # Add newline before and after
            desc = f"\n{desc}\n"
        return f"{self.__class__.__qualname__}({desc})"


OperationInvT = TypeVar("OperationInvT", bound=Operation)


@dataclass
class _BlockOpsIterator(Iterator[Operation]):
    """
    Single-pass iterable of the operations in a block. Follows the next_op for
    each operation.
    """

    next_op: Operation | None

    def __iter__(self):
        return self

    def __next__(self):
        next_op = self.next_op
        if next_op is None:
            raise StopIteration
        self.next_op = next_op.next_op
        return next_op


@dataclass
class _BlockOpsReverseIterator(Iterator[Operation]):
    """
    Single-pass iterable of the operations in a block. Follows the prev_op for
    each operation.
    """

    prev_op: Operation | None

    def __iter__(self):
        return self

    def __next__(self):
        prev_op = self.prev_op
        if prev_op is None:
            raise StopIteration
        self.prev_op = prev_op.prev_op
        return prev_op


@dataclass
class BlockOps(Reversible[Operation], Iterable[Operation]):
    """
    Multi-pass iterable of the operations in a block. Follows the next_op for
    each operation.
    """

    block: Block

    def __iter__(self):
        return _BlockOpsIterator(self.first)

    def __len__(self):
        result = 0
        for _ in self:
            result += 1
        return result

    def __bool__(self) -> bool:
        """Returns `True` if there are operations in this block."""
        return not self.block.is_empty

    def __reversed__(self):
        return _BlockOpsReverseIterator(self.block.last_op)

    @property
    def first(self) -> Operation | None:
        """
        First operation in the block, None if block is empty.
        """
        return self.block.first_op

    @property
    def last(self) -> Operation | None:
        """
        Last operation in the block, None if block is empty.
        """
        return self.block.last_op


@dataclass(init=False, eq=False, unsafe_hash=False)
class Block(_IRNode, IRWithUses, IRWithName):
    """A sequence of operations"""

    _args: tuple[BlockArgument, ...]
    """The basic block arguments."""

    _first_op: Operation | None = field(repr=False)
    _last_op: Operation | None = field(repr=False)

    _next_block: Block | None = field(default=None, repr=False)
    _prev_block: Block | None = field(default=None, repr=False)

    parent: Region | None = field(default=None, repr=False)
    """Parent region containing the block."""

    @staticmethod
    def is_default_block_name(name: str) -> bool:
        """Check if a name matches the default block naming pattern (bb followed by digits)."""
        return name.startswith("bb") and name[2:].isdigit() and name[2:] != ""

    def __init__(
        self,
        ops: Iterable[Operation] = (),
        *,
        arg_types: Iterable[Attribute] = (),
    ):
        super().__init__()
        self._args = tuple(
            BlockArgument(arg_type, self, index)
            for index, arg_type in enumerate(arg_types)
        )
        self._first_op = None
        self._last_op = None

        self.add_ops(ops)

    @property
    def arg_types(self) -> Sequence[Attribute]:
        return tuple(arg.type for arg in self._args)

    @property
    def parent_node(self) -> IRNode | None:
        return self.parent

    @property
    def ops(self) -> BlockOps:
        """Returns a multi-pass Iterable of this block's operations."""
        return BlockOps(self)

    @property
    def next_block(self) -> Block | None:
        """The next block in the parent region"""
        return self._next_block

    @property
    def prev_block(self) -> Block | None:
        """The previous block in the parent region"""
        return self._prev_block

    def predecessors(self) -> tuple[Block, ...]:
        return tuple(
            p for use in self.uses if (p := use.operation.parent_block()) is not None
        )

    def parent_op(self) -> Operation | None:
        return self.parent.parent if self.parent else None

    def parent_region(self) -> Region | None:
        return self.parent

    def parent_block(self) -> Block | None:
        return self.parent.parent.parent if self.parent and self.parent.parent else None

    def __repr__(self) -> str:
        return f"<Block {id(self)}(_args={repr(self._args)}, num_ops={len(self.ops)})>"

    @property
    def args(self) -> tuple[BlockArgument, ...]:
        """Returns the block arguments."""
        return self._args

    def insert_arg(self, arg_type: Attribute, index: int) -> BlockArgument:
        """
        Insert a new argument with a given type to the arguments list at a specific index.
        Returns the new argument.
        """
        if index < 0 or index > len(self._args):
            raise ValueError("Unexpected index")
        new_arg = BlockArgument(arg_type, self, index)
        for arg in self._args[index:]:
            arg.index += 1
        self._args = tuple(chain(self._args[:index], [new_arg], self._args[index:]))
        return new_arg

    def erase_arg(self, arg: BlockArgument, safe_erase: bool = True) -> None:
        """
        Erase a block argument.
        If safe_erase is True, check that the block argument is not used.
        If safe_erase is False, replace the block argument uses with an ErasedSSAVAlue.
        """
        if arg.block is not self:
            raise ValueError("Attempting to delete an argument of the wrong block")
        for block_arg in self._args[arg.index + 1 :]:
            block_arg.index -= 1
        self._args = tuple(chain(self._args[: arg.index], self._args[arg.index + 1 :]))
        arg.erase(safe_erase=safe_erase)

    def _attach_op(self, operation: Operation) -> None:
        """Attach an operation to the block, and check that it has no parents."""
        if operation.parent:
            raise ValueError(
                "Can't add to a block an operation already attached to a block."
            )
        if operation.is_ancestor(self):
            raise ValueError(
                "Can't add an operation to a block contained in the operation."
            )
        operation.parent = self

    @property
    def is_empty(self) -> bool:
        """Returns `True` if there are no operations in this block."""
        return self._first_op is None

    @property
    def first_op(self) -> Operation | None:
        """The first operation in this block."""
        return self._first_op

    @property
    def last_op(self) -> Operation | None:
        """The last operation in this block."""
        return self._last_op

    def insert_op_after(self, new_op: Operation, existing_op: Operation) -> None:
        """
        Inserts `new_op` into this block, after `existing_op`.
        `new_op` should not be attached to a block.
        """
        if existing_op.parent is not self:
            raise ValueError(
                "Can't insert operation after operation not in this block."
            )

        self._attach_op(new_op)

        next_op = existing_op.next_op
        existing_op._insert_next_op(new_op)  # pyright: ignore[reportPrivateUsage]
        if next_op is None:
            # No `next_op`, means `prev_op` is the last op in the block.
            self._last_op = new_op

    def insert_op_before(self, new_op: Operation, existing_op: Operation) -> None:
        """
        Inserts `new_op` into this block, before `existing_op`.
        `new_op` should not be attached to a block.
        """
        if existing_op.parent is not self:
            raise ValueError(
                "Can't insert operation before operation not in current block"
            )

        self._attach_op(new_op)

        prev_op = existing_op.prev_op
        existing_op._insert_prev_op(new_op)  # pyright: ignore[reportPrivateUsage]
        if prev_op is None:
            # No `prev_op`, means `next_op` is the first op in the block.
            self._first_op = new_op

    def add_op(self, operation: Operation) -> None:
        """
        Add an operation at the end of the block.
        The operation should not be attached to another block already.
        """
        if self._last_op is None:
            self._attach_op(operation)
            self._first_op = operation
            self._last_op = operation
        else:
            self.insert_op_after(operation, self._last_op)

    def add_ops(self, ops: Iterable[Operation]) -> None:
        """
        Add operations at the end of the block.
        The operations should not be attached to another block.
        """
        for op in ops:
            self.add_op(op)

    def insert_ops_before(
        self, ops: Sequence[Operation], existing_op: Operation
    ) -> None:
        for op in ops:
            self.insert_op_before(op, existing_op)

    def insert_ops_after(
        self, ops: Sequence[Operation], existing_op: Operation
    ) -> None:
        for op in ops:
            self.insert_op_after(op, existing_op)

            existing_op = op

    def split_before(
        self,
        b_first: Operation,
        *,
        arg_types: Iterable[Attribute] = (),
    ) -> Block:
        """
        Split the block into two blocks before the specified operation.

        Note that all operations before the one given stay as part of the original basic
        block, and the rest of the operations in the original block are moved to the new
        block, including the old terminator.
        The original block is left without a terminator.
        The newly formed block is inserted into the parent region immediately after `self`
        and returned.
        """
        # Use `a` for new contents of `self`, and `b` for new block.
        if b_first.parent is not self:
            raise ValueError("Cannot split block on operation outside of the block.")

        parent = self.parent
        if parent is None:
            raise ValueError("Cannot split block with no parent.")

        first_of_self = self._first_op
        assert first_of_self is not None

        last_of_self = self._last_op
        assert last_of_self is not None

        a_last = b_first.prev_op
        b_last = last_of_self
        if a_last is None:
            # `before` is the first op in the Block, so all the ops move to the new block
            a_first = None
        else:
            a_first = first_of_self

        # Update first and last ops of self
        self._first_op = a_first
        self._last_op = a_last

        b = Block(arg_types=arg_types)
        a_index = parent.get_block_index(self)
        parent.insert_block(b, a_index + 1)

        b._first_op = b_first
        b._last_op = b_last

        # Update parent for moved ops
        b_iter: Operation | None = b_first
        while b_iter is not None:
            b_iter.parent = b
            b_iter = b_iter.next_op

        # Update next op for self.last
        if a_last is not None:
            a_last._next_op = None  # pyright: ignore[reportPrivateUsage]

        # Update previous op for b.first
        b_first._prev_op = None  # pyright: ignore[reportPrivateUsage]

        return b

    def get_operation_index(self, op: Operation) -> int:
        """Get the operation position in a block."""
        if op.parent is not self:
            raise ValueError("Operation is not a child of the block.")
        return next(idx for idx, block_op in enumerate(self.ops) if block_op is op)

    def detach_op(self, op: Operation) -> Operation:
        """
        Detach an operation from the block.
        Returns the detached operation.
        """
        if op.parent is not self:
            raise ValueError("Cannot detach operation from a different block.")
        op.parent = None

        prev_op = op.prev_op
        next_op = op.next_op

        if prev_op is not None:
            # detach op from linked list
            prev_op._next_op = next_op  # pyright: ignore[reportPrivateUsage]
            # detach linked list from op
            op._prev_op = None  # pyright: ignore[reportPrivateUsage]
        else:
            # reattach linked list if op is first op this block
            assert self._first_op is op
            self._first_op = next_op

        if next_op is not None:
            # detach op from linked list
            next_op._prev_op = prev_op  # pyright: ignore[reportPrivateUsage]
            # detach linked list from op
            op._next_op = None  # pyright: ignore[reportPrivateUsage]
        else:
            # reattach linked list if op is last op in this block
            assert self._last_op is op
            self._last_op = prev_op

        return op

    def erase_op(self, op: Operation, safe_erase: bool = True) -> None:
        """
        Erase an operation from the block.
        If safe_erase is True, check that the operation has no uses.
        """
        op = self.detach_op(op)
        op.erase(safe_erase=safe_erase)

    def walk(
        self, *, reverse: bool = False, region_first: bool = False
    ) -> Iterable[Operation]:
        """
        Iterate over all operations contained in the block.
        If region_first is set, then the operation regions are iterated before the
        operation. If reverse is set, then the region, block, and operation lists are
        iterated in reverse order.
        """
        for op in reversed(self.ops) if reverse else self.ops:
            yield from op.walk(reverse=reverse, region_first=region_first)

    def walk_blocks(self, *, reverse: bool = False) -> Iterator[Block]:
        """
        Iterate over all the blocks nested within this block, including self, in the
        order in which they are printed in the IR.
        Iterate in reverse order if reverse is True.
        """
        if not reverse:
            yield self
        for op in reversed(self.ops) if reverse else self.ops:
            yield from op.walk_blocks(reverse=reverse)
        if reverse:
            yield self

    def verify(self) -> None:
        for operation in self.ops:
            if operation.parent != self:
                raise ValueError(
                    "Parent pointer of operation does not refer to containing region"
                )
            operation.verify()

        if len(self.ops) == 0:
            if (region_parent := self.parent) is not None and (
                parent_op := region_parent.parent
            ) is not None:
                if len(region_parent.blocks) == 1 and not parent_op.has_trait(
                    NoTerminator
                ):
                    raise VerifyException(
                        f"Operation {parent_op.name} contains empty block in "
                        "single-block region that expects at least a terminator"
                    )

    def drop_all_references(self) -> None:
        """
        Drop all references to other operations.
        This function is called prior to deleting a block.
        """
        self.parent = None
        self._next_block = None
        self._prev_block = None
        for op in self.ops:
            op.drop_all_references()

    def find_ancestor_op_in_block(self, op: Operation) -> Operation | None:
        """
        Traverse up the operation hierarchy starting from op to find the ancestor
        operation that resides in the block.

        Returns None if no ancestor is found.
        """
        curr_op = op
        while curr_op.parent_block() != self:
            if (curr_op := curr_op.parent_op()) is None:
                return None

        return curr_op

    def erase(self, safe_erase: bool = True) -> None:
        """
        Erase the block, and remove all its references to other operations.
        If safe_erase is specified, check that no operation results are used outside
        the block.
        """
        assert self.parent is None, (
            "Blocks with parents should first be detached " + "before erasure."
        )
        self.drop_all_references()
        for op in self.ops:
            op.erase(safe_erase=safe_erase, drop_references=False)

    def is_structurally_equivalent(
        self,
        other: IRNode,
        context: dict[IRNode | SSAValue, IRNode | SSAValue] | None = None,
    ) -> bool:
        """
        Check if two blocks are structurally equivalent.
        The context is a mapping of IR nodes to IR nodes that are already known
        to be equivalent. This enables checking whether the use dependencies and
        successors are equivalent.
        """
        if context is None:
            context = {}
        if not isinstance(other, Block):
            return False
        if len(self.args) != len(other.args) or len(self.ops) != len(other.ops):
            return False
        for arg, other_arg in zip(self.args, other.args):
            if arg.type != other_arg.type:
                return False
            context[arg] = other_arg
        # Add self to the context so Operations can check for identical parents
        context[self] = other
        if not all(
            op.is_structurally_equivalent(other_op, context)
            for op, other_op in zip(self.ops, other.ops)
        ):
            return False

        return True


@dataclass
class _RegionBlocksIterator(Iterator[Block]):
    """
    Single-pass iterable of the blocks in a region. Follows the next_block for
    each operation.
    """

    next_block: Block | None

    def __iter__(self):
        return self

    def __next__(self):
        next_block = self.next_block
        if next_block is None:
            raise StopIteration
        self.next_block = next_block.next_block
        return next_block


@dataclass
class OpSuccessors(Sequence[Block]):
    """
    A view of the successor list of an operation.
    Any modification to the view is reflected on the operation.
    """

    _op: Operation
    """The operation owning the successors."""

    @overload
    def __getitem__(self, idx: int) -> Block: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[Block]: ...

    def __getitem__(self, idx: int | slice) -> Block | Sequence[Block]:
        return self._op._successors[idx]  # pyright: ignore[reportPrivateUsage]

    def __setitem__(self, idx: int, successor: Block) -> None:
        successors = self._op._successors  # pyright: ignore[reportPrivateUsage]
        successor_uses = self._op._successor_uses  # pyright: ignore[reportPrivateUsage]
        successors[idx].remove_use(successor_uses[idx])
        successor.add_use(successor_uses[idx])
        new_successors = (*successors[:idx], successor, *successors[idx + 1 :])
        self._op._successors = new_successors  # pyright: ignore[reportPrivateUsage]

    def __iter__(self) -> Iterator[Block]:
        return iter(self._op._successors)  # pyright: ignore[reportPrivateUsage]

    def __len__(self) -> int:
        return len(self._op._successors)  # pyright: ignore[reportPrivateUsage]

    def __eq__(self, other: object):
        if not isinstance(other, OpSuccessors):
            return False
        return (
            self._op._successors  # pyright: ignore[reportPrivateUsage]
            == other._op._successors  # pyright: ignore[reportPrivateUsage]
        )

    def __hash__(self):
        return hash(self._op._successors)  # pyright: ignore[reportPrivateUsage]


@dataclass
class _RegionBlocksReverseIterator(Iterator[Block]):
    """
    Single-pass iterable of the blocks in a region. Follows the prev_block for
    each block.
    """

    prev_block: Block | None

    def __iter__(self):
        return self

    def __next__(self):
        prev_block = self.prev_block
        if prev_block is None:
            raise StopIteration
        self.prev_block = prev_block.prev_block
        return prev_block


@dataclass
class RegionBlocks(Sequence[Block], Reversible[Block]):
    """
    Multi-pass iterable of the blocks in a region.
    """

    _region: Region

    def __iter__(self):
        return _RegionBlocksIterator(self._region.first_block)

    @overload
    def __getitem__(self, idx: int) -> Block: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[Block]: ...

    def __getitem__(self, idx: int | slice) -> Block | Sequence[Block]:
        if isinstance(idx, int):
            if 0 <= idx:
                for i, b in enumerate(self):
                    if i == idx:
                        return b
                raise IndexError
            else:
                for i, b in enumerate(reversed(self)):
                    if -1 == i + idx:
                        return b
                raise IndexError
        else:
            # This is possible but would require a bit of work to handle complex slices
            raise NotImplementedError("Indexing of RegionBlocks not yet implemented")

    def __len__(self):
        i = 0
        for _ in self:
            i += 1
        return i

    def __bool__(self) -> bool:
        """Returns `True` if there are blocks in this region."""
        first_block = self._region.first_block
        return first_block is not None

    def __reversed__(self):
        return _RegionBlocksReverseIterator(self._region.last_block)

    @property
    def first(self) -> Block | None:
        """
        First block in the region, None if region is empty.
        """
        return self._region.first_block

    @property
    def last(self) -> Block | None:
        """
        Last block in the region, None if region is empty.
        """
        return self._region.last_block


@dataclass(init=False, eq=False, unsafe_hash=False)
class Region(_IRNode):
    """A region contains a CFG of blocks. Regions are contained in operations."""

    class DEFAULT:
        """
        A marker to be used as a default parameter to functions when a default
        single-block region should be constructed.
        """

    _first_block: Block | None = field(default=None, repr=False)
    """The first block in the region. This is the entry block if it is present."""

    _last_block: Block | None = field(default=None, repr=False)
    """The last block in the region."""

    parent: Operation | None = field(default=None, repr=False)
    """Operation containing the region."""

    def __init__(self, blocks: Block | Iterable[Block] = ()):
        super().__init__()
        self.add_block(blocks)

    @property
    def parent_node(self) -> IRNode | None:
        return self.parent

    def parent_block(self) -> Block | None:
        return self.parent.parent if self.parent else None

    def parent_op(self) -> Operation | None:
        return self.parent

    def parent_region(self) -> Region | None:
        return (
            self.parent.parent.parent
            if self.parent is not None and self.parent.parent is not None
            else None
        )

    def find_ancestor_block_in_region(self, block: Block) -> Block | None:
        """
        Returns 'block' if 'block' lies in this region, or otherwise finds
        the ancestor of 'block' that lies in this region.

        Returns None if no ancestor block that lies in this region is found.
        """
        curr_block = block
        while curr_block.parent_region() != self:
            curr_block = curr_block.parent_block()
            if curr_block is None:
                return None

        return curr_block

    @property
    def blocks(self) -> RegionBlocks:
        """
        A multi-pass iterable of blocks.
        """
        return RegionBlocks(self)

    @property
    def first_block(self) -> Block | None:
        """First block in this region. This is the entry block if present."""
        return self._first_block

    @property
    def last_block(self) -> Block | None:
        """Last block in this region."""
        return self._last_block

    def __repr__(self) -> str:
        short_reprs = ", ".join(_short_repr(block) for block in self.blocks)
        return f"Region(blocks=[{short_reprs}])"

    @property
    def ops(self) -> BlockOps:
        """
        Get the operations of a single-block region.
        Returns an exception if the region is not single-block.
        """
        if len(self.blocks) != 1:
            raise ValueError(
                "'ops' property of Region class is only available "
                "for single-block regions."
            )
        return self.block.ops

    @property
    def op(self) -> Operation:
        """
        Get the operation of a single-operation single-block region.
        Returns an exception if the region is not single-operation single-block.
        """
        if len(self.blocks) == 1:
            block = self.block
            first_op = block.first_op
            last_op = block.last_op
            if first_op is last_op and first_op is not None:
                return first_op
        raise ValueError(
            "'op' property of Region class is only available "
            "for single-operation single-block regions."
        )

    @property
    def block(self) -> Block:
        """
        Get the block of a single-block region.
        Returns an exception if the region is not single-block.
        """
        if self._first_block is None or self._first_block is not self._last_block:
            raise ValueError(
                "'block' property of Region class is only available "
                "for single-block regions."
            )
        return self._first_block

    def _attach_block(self, block: Block) -> None:
        """Attach a block to the region, and check that it has no parents."""
        if block.parent:
            raise ValueError(
                "Can't add to a region a block already attached to a region."
            )
        if block.is_ancestor(self):
            raise ValueError("Can't add a block to a region contained in the block.")
        block.parent = self

    def add_block(self, block: Block | Iterable[Block]) -> None:
        """
        Insert one or multiple blocks at the end of the region.
        The blocks should not be attached to another region.
        """
        blocks_iter: Iterator[Block]
        if isinstance(block, Block):
            blocks_iter = iter((block,))
        else:
            blocks_iter = iter(block)
        prev_block = self.last_block

        if prev_block is None:
            try:
                # First block
                prev_block = next(blocks_iter)
                self._attach_block(prev_block)
                self._first_block = prev_block
            except StopIteration:
                # blocks_iter is empty, nothing to do
                return

        try:
            while True:
                next_block = next(blocks_iter)
                self._attach_block(next_block)
                next_block._prev_block = (  # pyright: ignore[reportPrivateUsage]
                    prev_block
                )
                prev_block._next_block = (  # pyright: ignore[reportPrivateUsage]
                    next_block
                )
                prev_block = next_block

        except StopIteration:
            # Repair last block
            self._last_block = prev_block
            return

    def insert_block_before(
        self, block: Block | Iterable[Block], target: Block
    ) -> None:
        """
        Insert one or multiple blocks before a given block in the region.
        The blocks should not be attached to another region.
        """
        if target.parent is not self:
            raise ValueError(
                "Cannot insert blocks before a block into a region that is not the target's parent"
            )
        blocks_iter: Iterator[Block]
        if isinstance(block, Block):
            blocks_iter = iter((block,))
        else:
            blocks_iter = iter(block)
        prev_block = target.prev_block

        if prev_block is None:
            try:
                # First block
                new_first = next(blocks_iter)
                self._attach_block(new_first)
                self._first_block = new_first
                new_first._next_block = target  # pyright: ignore[reportPrivateUsage]
                prev_block = new_first
            except StopIteration:
                # blocks_iter is empty, nothing to do
                return

        # The invariant for the loop is that prev_block is always before target when
        # calling `next`.

        try:
            while True:
                next_block = next(blocks_iter)
                self._attach_block(next_block)
                next_block._prev_block = (  # pyright: ignore[reportPrivateUsage]
                    prev_block
                )
                prev_block._next_block = (  # pyright: ignore[reportPrivateUsage]
                    next_block
                )
                prev_block = next_block

        except StopIteration:
            # Repair broken link
            prev_block._next_block = target  # pyright: ignore[reportPrivateUsage]
            target._prev_block = prev_block  # pyright: ignore[reportPrivateUsage]
            return

    def insert_block_after(self, block: Block | Iterable[Block], target: Block) -> None:
        """
        Insert one or multiple blocks after a given block in the region.
        The blocks should not be attached to another region.
        """
        next_block = target.next_block
        if next_block is None:
            self.add_block(block)
        else:
            self.insert_block_before(block, next_block)

    def insert_block(self, blocks: Block | Iterable[Block], index: int) -> None:
        """
        Insert one or multiple blocks at a given index in the region.
        The blocks should not be attached to another region.
        """
        i = -1
        for i, b in enumerate(self.blocks):
            if i == index:
                self.insert_block_before(blocks, b)
                return
        if i + 1 == index:
            # Append block
            self.add_block(blocks)

    def get_block_index(self, block: Block) -> int:
        """Get the block position in a region."""
        if block.parent is not self:
            raise ValueError("Block is not a child of the region.")
        return next(
            idx for idx, region_block in enumerate(self.blocks) if region_block is block
        )

    def detach_block(self, block: int | Block) -> Block:
        """
        Detach a block from the region.
        Returns the detached block.
        """
        if isinstance(block, int):
            block = self.blocks[block]
        else:
            if block.parent is not self:
                raise ValueError("Block is not a child of the region.")

        block.parent = None
        if (prev_block := block.prev_block) is None:
            self._first_block = block.next_block
        else:
            prev_block._next_block = (  # pyright: ignore[reportPrivateUsage]
                block.next_block
            )
        if (next_block := block.next_block) is None:
            self._last_block = block.prev_block
        else:
            next_block._prev_block = (  # pyright: ignore[reportPrivateUsage]
                block.prev_block
            )

        return block

    def erase_block(self, block: int | Block, safe_erase: bool = True) -> None:
        """
        Erase a block from the region.
        If safe_erase is True, check that the block has no uses.
        """
        block = self.detach_block(block)
        block.erase(safe_erase=safe_erase)

    def clone(self) -> Region:
        """
        Clone the entire region into a new one.
        """
        new_region = Region()
        self.clone_into(new_region)
        return new_region

    def clone_into(
        self,
        dest: Region,
        insert_index: int | None = None,
        value_mapper: dict[SSAValue, SSAValue] | None = None,
        block_mapper: dict[Block, Block] | None = None,
        *,
        clone_name_hints: bool = True,
        clone_operands: bool = True,
    ):
        """
        Clone all block of this region into `dest` to position `insert_index`
        """
        assert dest != self
        if insert_index is None:
            insert_index = len(dest.blocks)
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}

        new_blocks: list[Block] = []

        # Clone all blocks without their contents, and register the block mapping
        # This ensures that operations can refer to blocks that are not yet cloned
        for block in self.blocks:
            new_block = Block()
            new_blocks.append(new_block)
            block_mapper[block] = new_block

        dest.insert_block(new_blocks, insert_index)

        # Populate the blocks with the cloned operations
        for block, new_block in zip(self.blocks, new_blocks):
            for idx, block_arg in enumerate(block.args):
                new_block.insert_arg(block_arg.type, idx)
                new_arg = new_block.args[idx]
                value_mapper[block_arg] = new_arg
                if clone_name_hints:
                    new_arg.name_hint = block_arg.name_hint
            for op in block.ops:
                new_block.add_op(
                    op.clone(
                        value_mapper,
                        block_mapper,
                        clone_name_hints=clone_name_hints,
                        clone_operands=False,
                    )
                )
        # Handle cases where results may be created after their first use when walking
        # in lexicographic order.
        if clone_operands:
            for old, new in zip(self.walk(), dest.walk()):
                new.operands = tuple(
                    value_mapper.get(operand, operand) for operand in old.operands
                )

    def walk(
        self, *, reverse: bool = False, region_first: bool = False
    ) -> Iterator[Operation]:
        """
        Call a function on all operations contained in the region.
        If region_first is set, then the operation regions are iterated before the
        operation. If reverse is set, then the region, block, and operation lists are
        iterated in reverse order.
        """
        for block in reversed(self.blocks) if reverse else self.blocks:
            yield from block.walk(reverse=reverse, region_first=region_first)

    def verify(self) -> None:
        for block in self.blocks:
            block.verify()
            if block.parent != self:
                raise ValueError(
                    "Parent pointer of block does not refer to containing region"
                )

    def drop_all_references(self) -> None:
        """
        Drop all references to other operations.
        This function is called prior to deleting a region.
        """
        self.parent = None
        for block in self.blocks:
            block.drop_all_references()

    def erase(self) -> None:
        """
        Erase the region, and remove all its references to other operations.
        """
        assert self.parent, (
            "Regions with parents should first be " + "detached before erasure."
        )
        self.drop_all_references()

    def move_blocks(self, region: Region) -> None:
        """
        Move the blocks of this region to another region. Leave no blocks in this region.
        """
        if region is self:
            raise ValueError("Cannot move region into itself.")
        self_first_block = self._first_block
        if self_first_block is None:
            return
        self_last_block = self._last_block
        assert self_last_block is not None
        other_last_block = region.last_block
        if other_last_block is None:
            region._first_block = self._first_block
        else:
            self_first_block._prev_block = (  # pyright: ignore[reportPrivateUsage]
                other_last_block
            )
            other_last_block._next_block = (  # pyright: ignore[reportPrivateUsage]
                self_first_block
            )
        region._last_block = self_last_block

        for block in self.blocks:
            block.parent = region

        self._first_block = None
        self._last_block = None

    def move_blocks_before(self, target: Block) -> None:
        """
        Move the blocks of this region to another region, before the target block.
        Leave no blocks in this region.
        """
        region = target.parent
        if region is self:
            raise ValueError("Cannot move region into itself.")
        if region is None:
            raise ValueError("Cannot inline region before a block with no parent")

        first_block = self._first_block
        if not first_block:
            return
        last_block = self._last_block
        assert last_block is not None

        if target.prev_block is None:
            region._first_block = first_block
        else:
            target.prev_block._next_block = (  # pyright: ignore[reportPrivateUsage]
                first_block
            )
            first_block._prev_block = (  # pyright: ignore[reportPrivateUsage]
                target.prev_block
            )

        for block in self.blocks:
            block.parent = region

        last_block._next_block = target  # pyright: ignore[reportPrivateUsage]
        target._prev_block = last_block  # pyright: ignore[reportPrivateUsage]

        self._first_block = None
        self._last_block = None

    def is_structurally_equivalent(
        self,
        other: IRNode,
        context: dict[IRNode | SSAValue, IRNode | SSAValue] | None = None,
    ) -> bool:
        """
        Check if two regions are structurally equivalent.
        The context is a mapping of IR nodes to IR nodes that are already known
        to be equivalent. This enables checking whether the use dependencies and
        successors are equivalent.
        """
        if context is None:
            context = {}
        if not isinstance(other, Region):
            return False
        if len(self.blocks) != len(other.blocks):
            return False
        # register all blocks in the context so we can check whether ops have
        # the corrects successors
        for block, other_block in zip(self.blocks, other.blocks):
            context[block] = other_block
        if not all(
            block.is_structurally_equivalent(other_block, context)
            for block, other_block in zip(self.blocks, other.blocks)
        ):
            return False
        return True


IRNode: TypeAlias = Operation | Region | Block


def _short_repr(value: object) -> str:
    """
    Helper function to avoid recursion in reprs.
    """
    return repr(value) if value is None else f"<{value.__class__.__name__} {id(value)}>"
