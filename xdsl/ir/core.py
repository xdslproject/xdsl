from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    NoReturn,
    Protocol,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import Self

from xdsl.traits import IsTerminator, NoTerminator, OpTrait, OpTraitInvT
from xdsl.utils import lexer
from xdsl.utils.deprecation import deprecated
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.irdl import ParamAttrDef
    from xdsl.parser import AttrParser, Parser
    from xdsl.printer import Printer

OpT = TypeVar("OpT", bound="Operation")


@dataclass
class Dialect:
    """Contains the operations and attributes of a specific dialect"""

    _name: str

    _operations: list[type[Operation]] = field(
        default_factory=list, init=True, repr=True
    )
    _attributes: list[type[Attribute]] = field(
        default_factory=list, init=True, repr=True
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


@dataclass
class MLContext:
    """Contains structures for operations/attributes registration."""

    allow_unregistered: bool = field(default=False)

    _loaded_dialects: dict[str, Dialect] = field(default_factory=dict)
    _loaded_ops: dict[str, type[Operation]] = field(default_factory=dict)
    _loaded_attrs: dict[str, type[Attribute]] = field(default_factory=dict)
    _registered_dialects: dict[str, Callable[[], Dialect]] = field(default_factory=dict)
    """
    A dictionary of all registered dialects that are not yet loaded. This is used to
    only load the respective Python files when the dialect is actually used.
    """

    def clone(self) -> MLContext:
        return MLContext(
            self.allow_unregistered,
            self._loaded_dialects.copy(),
            self._loaded_ops.copy(),
            self._loaded_attrs.copy(),
            self._registered_dialects.copy(),
        )

    @property
    def loaded_ops(self) -> Iterable[type[Operation]]:
        """
        Returns all the loaded operations. Not valid across mutations of this object.
        """
        return self._loaded_ops.values()

    @property
    def loaded_attrs(self) -> Iterable[type[Attribute]]:
        """
        Returns all the loaded attributes. Not valid across mutations of this object.
        """
        return self._loaded_attrs.values()

    @property
    def loaded_dialects(self) -> Iterable[Dialect]:
        """
        Returns all the loaded attributes. Not valid across mutations of this object.
        """
        return self._loaded_dialects.values()

    @property
    def registered_dialect_names(self) -> Iterable[str]:
        """
        Returns the names of all registered dialects. Not valid across mutations of this object.
        """
        return self._registered_dialects.keys()

    def register_dialect(
        self, name: str, dialect_factory: Callable[[], Dialect]
    ) -> None:
        """
        Register a dialect without loading it. The dialect is only loaded in the context
        when an operation or attribute of that dialect is parsed, or when explicitely
        requested with `load_registered_dialect`.
        """
        if name in self._registered_dialects:
            raise ValueError(f"'{name}' dialect is already registered")
        self._registered_dialects[name] = dialect_factory

    def load_registered_dialect(self, name: str) -> None:
        """Load a dialect that is already registered in the context."""
        if name not in self._registered_dialects:
            raise ValueError(f"'{name}' dialect is not registered")
        dialect = self._registered_dialects[name]()
        self._loaded_dialects[dialect.name] = dialect

        for op in dialect.operations:
            self.load_op(op)

        for attr in dialect.attributes:
            self.load_attr(attr)

    def load_dialect(self, dialect: Dialect):
        """
        Load a dialect. Operation and Attribute names should be unique.
        If the dialect is already registered in the context, use
        `load_registered_dialect` instead.
        """
        if dialect.name in self._registered_dialects:
            raise ValueError(
                f"'{dialect.name}' dialect is already registered, use 'load_registered_dialect' instead"
            )
        self.register_dialect(dialect.name, lambda: dialect)
        self.load_registered_dialect(dialect.name)

    def load_op(self, op: type[Operation]) -> None:
        """Load an operation definition. Operation names should be unique."""
        if op.name in self._loaded_ops:
            raise Exception(f"Operation {op.name} has already been loaded")
        self._loaded_ops[op.name] = op

    def load_attr(self, attr: type[Attribute]) -> None:
        """Load an attribute definition. Attribute names should be unique."""
        if attr.name in self._loaded_attrs:
            raise Exception(f"Attribute {attr.name} has already been loaded")
        self._loaded_attrs[attr.name] = attr

    def get_optional_op(self, name: str) -> type[Operation] | None:
        """
        Get an operation class from its name if it exists.
        If the operation is not registered, return None unless unregistered operations
        are allowed in the context, in which case return an UnregisteredOp.
        """
        # If the operation is already loaded, returns it.
        if name in self._loaded_ops:
            return self._loaded_ops[name]

        # Otherwise, check if the operation dialect is registered.
        if "." in name:
            dialect_name, _ = name.split(".", 1)
            if dialect_name in self._loaded_dialects:
                return None
            if dialect_name in self._registered_dialects:
                self.load_registered_dialect(dialect_name)
                return self.get_optional_op(name)

        # If the dialect is unregistered, but the context allows unregistered
        # operations, return an UnregisteredOp.
        if self.allow_unregistered:
            from xdsl.dialects.builtin import UnregisteredOp

            op_type = UnregisteredOp.with_name(name)
            self._loaded_ops[name] = op_type
            return op_type
        return None

    def get_op(self, name: str) -> type[Operation]:
        """
        Get an operation class from its name.
        If the operation is not registered, raise an exception unless unregistered
        operations are allowed in the context, in which case return an UnregisteredOp.
        """
        if op_type := self.get_optional_op(name):
            return op_type
        raise Exception(f"Operation {name} is not registered")

    def get_optional_attr(
        self,
        name: str,
        create_unregistered_as_type: bool = False,
    ) -> type[Attribute] | None:
        """
        Get an attribute class from its name if it exists.
        If the attribute is not registered, return None unless unregistered attributes
        are allowed in the context, in which case return an UnregisteredAttr.
        Since UnregisteredAttr may be a type (for MLIR compatibility), an
        additional flag is required to create an UnregisterAttr that is
        also a type.
        """
        # If the attribute is already loaded, returns it.
        if name in self._loaded_attrs:
            return self._loaded_attrs[name]

        # Otherwise, check if the attribute dialect is registered.
        dialect_name, _ = name.split(".", 1)
        if dialect_name in self._registered_dialects:
            if dialect_name in self._loaded_dialects:
                return None
            self.load_registered_dialect(dialect_name)
            return self.get_optional_attr(name)

        # If the dialect is unregistered, but the context allows unregistered
        # attributes, return an UnregisteredOp.
        if self.allow_unregistered:
            from xdsl.dialects.builtin import UnregisteredAttr

            attr_type = UnregisteredAttr.with_name_and_type(
                name, create_unregistered_as_type
            )
            self._loaded_attrs[name] = attr_type
            return attr_type

        return None

    def get_attr(
        self,
        name: str,
        create_unregistered_as_type: bool = False,
    ) -> type[Attribute]:
        """
        Get an attribute class from its name.
        If the attribute is not registered, raise an exception unless unregistered
        attributes are allowed in the context, in which case return an UnregisteredAttr.
        Since UnregisteredAttr may be a type (for MLIR compatibility), an
        additional flag is required to create an UnregisterAttr that is
        also a type.
        """
        if attr_type := self.get_optional_attr(name, create_unregistered_as_type):
            return attr_type
        raise Exception(f"Attribute {name} is not registered")

    def get_dialect(self, name: str) -> Dialect:
        if (dialect := self.get_optional_dialect(name)) is None:
            raise Exception(f"Dialect {name} is not registered")
        return dialect

    def get_optional_dialect(self, name: str) -> Dialect | None:
        if name in self._loaded_dialects:
            return self._loaded_dialects[name]
        return None


@dataclass(frozen=True)
class Use:
    """The use of a SSA value."""

    operation: Operation
    """The operation using the value."""

    index: int
    """The index of the operand using the value in the operation."""


@dataclass
class SSAValue(ABC):
    """
    A reference to an SSA variable.
    An SSA variable is either an operation result, or a basic block argument.
    """

    type: Attribute
    """Each SSA variable is associated to a type."""

    uses: set[Use] = field(init=False, default_factory=set, repr=False)
    """All uses of the value."""

    _name: str | None = field(init=False, default=None)

    _name_regex: ClassVar[re.Pattern[str]] = re.compile(r"([A-Za-z_$.-][\w$.-]*)")

    @property
    @abstractmethod
    def owner(self) -> Operation | Block:
        """
        An SSA variable is either an operation result, or a basic block argument.
        This property returns the Operation or Block that currently defines a specific value.
        """
        pass

    @property
    def name_hint(self) -> str | None:
        return self._name

    @name_hint.setter
    def name_hint(self, name: str | None):
        # only allow valid names
        if SSAValue.is_valid_name(name):
            self._name = name
        else:
            raise ValueError(
                "Invalid SSA Value name format!",
                r"Make sure names contain only characters of [A-Za-z0-9_$.-] and don't start with a number!",
            )

    @classmethod
    def is_valid_name(cls, name: str | None):
        return name is None or cls._name_regex.fullmatch(name)

    @staticmethod
    def get(arg: SSAValue | Operation) -> SSAValue:
        "Get a new SSAValue from either a SSAValue, or an operation with a single result."
        match arg:
            case SSAValue():
                return arg
            case Operation():
                if len(arg.results) == 1:
                    return arg.results[0]
                raise ValueError(
                    "SSAValue.build: expected operation with a single result."
                )

    def add_use(self, use: Use):
        """Add a new use of the value."""
        self.uses.add(use)

    def remove_use(self, use: Use):
        """Remove a use of the value."""
        assert use in self.uses, "use to be removed was not in use list"
        self.uses.remove(use)

    def replace_by(self, value: SSAValue) -> None:
        """Replace the value by another value in all its uses."""
        for use in self.uses.copy():
            use.operation.operands[use.index] = value
        # carry over name if possible
        if value.name_hint is None:
            value.name_hint = self.name_hint
        assert len(self.uses) == 0, "unexpected error in xdsl"

    def erase(self, safe_erase: bool = True) -> None:
        """
        Erase the value.
        If safe_erase is True, then check that no operations use the value anymore.
        If safe_erase is False, then replace its uses by an ErasedSSAValue.
        """
        if safe_erase and len(self.uses) != 0:
            raise Exception(
                "Attempting to delete SSA value that still has uses of result "
                f"of operation:\n{self.owner}"
            )
        self.replace_by(ErasedSSAValue(self.type, self))


@dataclass
class OpResult(SSAValue):
    """A reference to an SSA variable defined by an operation result."""

    op: Operation
    """The operation defining the variable."""

    index: int
    """The index of the result in the defining operation."""

    @property
    def owner(self) -> Operation:
        return self.op

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.type}] index: {self.index}, operation: {self.op.name}, uses: {len(self.uses)}>"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class BlockArgument(SSAValue):
    """A reference to an SSA variable defined by a basic block argument."""

    block: Block
    """The block defining the variable."""

    index: int
    """The index of the variable in the block arguments."""

    @property
    def owner(self) -> Block:
        return self.block

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.type}] index: {self.index}, uses: {len(self.uses)}>"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class ErasedSSAValue(SSAValue):
    """
    An erased SSA variable.
    This is used during transformations when a SSA variable is destroyed but still used.
    """

    old_value: SSAValue

    @property
    def owner(self) -> Operation | Block:
        return self.old_value.owner

    def __hash__(self) -> int:
        return hash(id(self))


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
        Raise an exception otherwise.
        """
        pass

    def __str__(self) -> str:
        from xdsl.printer import Printer

        res = StringIO()
        printer = Printer(stream=res)
        printer.print_attribute(self)
        return res.getvalue()


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
    This class is only used for printing attributes in the opaque form,
    as described at https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values.
    """

    pass


DataElement = TypeVar("DataElement", covariant=True)

AttributeCovT = TypeVar("AttributeCovT", bound=Attribute, covariant=True)
AttributeInvT = TypeVar("AttributeInvT", bound=Attribute)


@dataclass(frozen=True)
class Data(Generic[DataElement], Attribute, ABC):
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
    @abstractmethod
    def parse_parameter(cls, parser: AttrParser) -> DataElement:
        """Parse the attribute parameter."""

    @abstractmethod
    def print_parameter(self, printer: Printer) -> None:
        """Print the attribute parameter."""


EnumType = TypeVar("EnumType", bound=StrEnum)


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

    class MyEnumAttribute(EnumAttribute[MyEnum], OpaqueSyntaxAttribute):
        name = "example.my_enum"
    ```
    To use this attribute suffices to have a textual representation
    of `example<my_enum first>` and ``example<my_enum second>``

    """

    enum_type: ClassVar[type[StrEnum]]

    def __init_subclass__(cls) -> None:
        """
        This hook first checks two constraints, enforced to keep the implementation
        reasonable, until more complex use cases appear. It then stores the Enum type
        used by the subclass to use in parsing/printing.

        The constraints are:

        - Only direct, specialized inheritance is allowed. That is, using a subclass
        of EnumAttribute as a base class is *not supported*.
          This simplifies type-hacking code and I don't see it being too restrictive
          anytime soon.
        - The StrEnum values must all be parsable as identifiers. This is to keep the
        parsing code simple and efficient. This restriction is easier to lift, but I
        haven't yet met an example use case where it matters, so I'm keeping it simple.
        """
        orig_bases = getattr(cls, "__orig_bases__")
        enumattr = next(b for b in orig_bases if get_origin(b) is EnumAttribute)
        enum_type = get_args(enumattr)[0]
        if isinstance(enum_type, TypeVar):
            raise TypeError("Only direct inheritance from EnumAttribute is allowed.")

        for v in enum_type:
            if lexer.Lexer.bare_identifier_suffix_regex.fullmatch(v) is None:
                raise ValueError(
                    "All StrEnum values of an EnumAttribute must be parsable as an identifer."
                )

        cls.enum_type = enum_type

    @final
    def print_parameter(self, printer: Printer) -> None:
        printer.print(" ", self.data.value)

    @final
    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> EnumType:
        enum_type = cls.enum_type

        val = parser.parse_identifier()
        if val not in enum_type.__members__.values():
            enum_values = list(enum_type)
            if len(enum_values) == 1:
                parser.raise_error(f"Expected `{enum_values[0]}`.")
            parser.raise_error(
                f"Expected `{'`, `'.join(enum_values[:-1])}` or `{enum_values[-1]}`."
            )
        return cast(EnumType, enum_type(val))


@dataclass(frozen=True, init=False)
class ParametrizedAttribute(Attribute):
    """An attribute parametrized by other attributes."""

    parameters: tuple[Attribute, ...] = field()

    def __init__(self, parameters: Sequence[Attribute] = ()):
        object.__setattr__(self, "parameters", tuple(parameters))
        super().__init__()

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

        # Call the __init__ of ParametrizedAttribute, which will set the
        # parameters field.
        ParametrizedAttribute.__init__(attr, tuple(params))
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


@dataclass(init=False)
class IRNode(ABC):
    def is_ancestor(self, op: IRNode) -> bool:
        "Returns true if the IRNode is an ancestor of another IRNode."
        if op is self:
            return True
        if (parent := op.parent_node) is None:
            return False
        return self.is_ancestor(parent)

    def get_toplevel_object(self) -> IRNode:
        """Get the operation, block, or region ancestor that has no parents."""
        if (parent := self.parent_node) is None:
            return self
        return parent.get_toplevel_object()

    def is_structurally_equivalent(
        self,
        other: IRNode,
        context: dict[IRNode | SSAValue, IRNode | SSAValue] | None = None,
    ) -> bool:
        """Check if two IR nodes are structurally equivalent."""
        ...

    @property
    @abstractmethod
    def parent_node(self) -> IRNode | None:
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...


@dataclass
class OpOperands(Sequence[SSAValue]):
    """
    A view of the operand list of an operation.
    Any modification to the view is reflected on the operation.
    """

    _op: Operation
    """The operation owning the operands."""

    @overload
    def __getitem__(self, idx: int) -> SSAValue:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[SSAValue]:
        ...

    def __getitem__(self, idx: int | slice) -> SSAValue | Sequence[SSAValue]:
        return self._op._operands[idx]  # pyright: ignore[reportPrivateUsage]

    def __setitem__(self, idx: int, operand: SSAValue) -> None:
        operands = self._op._operands  # pyright: ignore[reportPrivateUsage]
        operands[idx].remove_use(Use(self._op, idx))
        operand.add_use(Use(self._op, idx))
        new_operands = (*operands[:idx], operand, *operands[idx + 1 :])
        self._op._operands = new_operands  # pyright: ignore[reportPrivateUsage]

    def __iter__(self) -> Iterator[SSAValue]:
        return iter(self._op._operands)  # pyright: ignore[reportPrivateUsage]

    def __len__(self) -> int:
        return len(self._op._operands)  # pyright: ignore[reportPrivateUsage]


@dataclass
class Operation(IRNode):
    """A generic operation. Operation definitions inherit this class."""

    name: ClassVar[str] = field(repr=False)
    """The operation name. Should be a static member of the class"""

    _operands: tuple[SSAValue, ...] = field(default=())
    """The operation operands."""

    results: list[OpResult] = field(default_factory=list)
    """The results created by the operation."""

    successors: list[Block] = field(default_factory=list)
    """
    The basic blocks that the operation may give control to.
    This list should be empty for non-terminator operations.
    """

    properties: dict[str, Attribute] = field(default_factory=dict)
    """
    The properties attached to the operation.
    Properties are inherent to the definition of an operation's semantics, and
    thus cannot be discarded by transformations.
    """

    attributes: dict[str, Attribute] = field(default_factory=dict)
    """The attributes attached to the operation."""

    regions: list[Region] = field(default_factory=list)
    """Regions arguments of the operation."""

    parent: Block | None = field(default=None, repr=False)
    """The block containing this operation."""

    _next_op: Operation | None = field(default=None, repr=False)
    """Next operation in block containing this operation."""

    _prev_op: Operation | None = field(default=None, repr=False)
    """Previous operation in block containing this operation."""

    traits: ClassVar[frozenset[OpTrait]]
    """
    Traits attached to an operation definition.
    This is a static field, and is made empty by default by PyRDL if not set
    by the operation definition.
    """

    @property
    def parent_node(self) -> IRNode | None:
        return self.parent

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
        new = tuple(new)
        for idx, operand in enumerate(self._operands):
            operand.remove_use(Use(self, idx))
        for idx, operand in enumerate(new):
            operand.add_use(Use(self, idx))
        self._operands = new

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

        self.results = [
            OpResult(result_type, self, idx)
            for (idx, result_type) in enumerate(result_types)
        ]
        self.properties = dict(properties)
        self.attributes = dict(attributes)
        self.successors = list(successors)
        self.regions = []
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
            raise Exception(
                "Cannot add region that is already attached on an operation."
            )
        self.regions.append(region)
        region.parent = self

    def get_region_index(self, region: Region) -> int:
        """Get the region position in the operation."""
        if region.parent is not self:
            raise Exception("Region is not attached to the operation.")
        for idx, curr_region in enumerate(self.regions):
            if curr_region is region:
                return idx
        assert (
            False
        ), "The IR is corrupted. Operation seems to be the region's parent but still doesn't have the region attached to it."

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
        for idx, operand in enumerate(self.operands):
            operand.remove_use(Use(self, idx))
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

    @deprecated("Use walk(reverse=True, region_first=True) instead")
    def walk_reverse(self) -> Iterator[Operation]:
        """
        Iterate all operations contained in the operation (including this one) in reverse order.
        """
        for region in reversed(self.regions):
            yield from region.walk_reverse()
        yield self

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

    def verify(self, verify_nested_ops: bool = True) -> None:
        for operand in self.operands:
            if isinstance(operand, ErasedSSAValue):
                raise Exception("Erased SSA value is used by the operation")

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
                "Operation does not verify: " + str(err), underlying_error=err
            )

    def verify_(self) -> None:
        pass

    _OperationType = TypeVar("_OperationType", bound="Operation")

    @classmethod
    def parse(cls: type[_OperationType], parser: Parser) -> _OperationType:
        parser.raise_error(f"Operation {cls.name} does not have a custom format.")

    def print(self, printer: Printer):
        return printer.print_op_with_default_format(self)

    def clone_without_regions(
        self: OpT,
        value_mapper: dict[SSAValue, SSAValue] | None = None,
        block_mapper: dict[Block, Block] | None = None,
    ) -> OpT:
        """Clone an operation, with empty regions instead."""
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}
        operands = [
            (value_mapper[operand] if operand in value_mapper else operand)
            for operand in self.operands
        ]
        result_types = [res.type for res in self.results]
        attributes = self.attributes.copy()
        properties = self.properties.copy()
        successors = [
            (block_mapper[successor] if successor in block_mapper else successor)
            for successor in self.successors
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
        for idx, result in enumerate(cloned_op.results):
            value_mapper[self.results[idx]] = result
        return cloned_op

    def clone(
        self: OpT,
        value_mapper: dict[SSAValue, SSAValue] | None = None,
        block_mapper: dict[Block, Block] | None = None,
    ) -> OpT:
        """Clone an operation with all its regions and operations in them."""
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}
        op = self.clone_without_regions(value_mapper, block_mapper)
        for idx, region in enumerate(self.regions):
            region.clone_into(op.regions[idx], 0, value_mapper, block_mapper)
        return op

    @classmethod
    def has_trait(
        cls,
        trait: type[OpTrait],
        parameters: Any = None,
        value_if_unregistered: bool = True,
    ) -> bool:
        """
        Check if the operation implements a trait with the given parameters.
        If the operation is not registered, return value_if_unregisteed instead.
        """

        from xdsl.dialects.builtin import UnregisteredOp

        if issubclass(cls, UnregisteredOp):
            return value_if_unregistered

        return cls.get_trait(trait, parameters) is not None

    @classmethod
    def get_trait(
        cls, trait: type[OpTraitInvT], parameters: Any = None
    ) -> OpTraitInvT | None:
        """
        Return a trait with the given type and parameters, if it exists.
        """
        for t in cls.traits:
            if isinstance(t, trait) and t.parameters == parameters:
                return t
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
            raise Exception("Cannot detach a toplevel operation.")
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
        exception_type: type[Exception] = VerifyException,
        underlying_error: Exception | None = None,
    ) -> NoReturn:
        """Emit an error with the given message."""
        from xdsl.utils.diagnostic import Diagnostic

        diagnostic = Diagnostic()
        diagnostic.add_message(self, message)
        diagnostic.raise_exception(message, self, exception_type, underlying_error)

    @classmethod
    def dialect_name(cls) -> str:
        return cls.name.split(".")[0]

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        from xdsl.printer import Printer

        res = StringIO()
        printer = Printer(stream=res)
        printer.print_op(self)
        return res.getvalue()

    def __format__(self, __format_spec: str) -> str:
        desc = str(self)
        if "\n" in desc:
            # Description is multi-line, indent each line
            desc = "\n".join("\t" + line for line in desc.splitlines())
            # Add newline before and after
            desc = f"\n{desc}\n"
        return f"{self.__class__.__qualname__}({desc})"


OperationInvT = TypeVar("OperationInvT", bound=Operation)


@dataclass
class _BlockOpsIterator:
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
class _BlockOpsReverseIterator:
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
class BlockOps:
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


@dataclass
class BlockReverseOps:
    """
    Multi-pass iterable of the operations in a block. Follows the prev_op for
    each operation.
    """

    block: Block

    def __iter__(self):
        return _BlockOpsReverseIterator(self.block.last_op)

    def __len__(self):
        result = 0
        for _ in self:
            result += 1
        return result


@dataclass(init=False)
class Block(IRNode):
    """A sequence of operations"""

    _args: tuple[BlockArgument, ...]
    """The basic block arguments."""

    _first_op: Operation | None = field(repr=False)
    _last_op: Operation | None = field(repr=False)

    parent: Region | None = field(default=None, repr=False)
    """Parent region containing the block."""

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
    def parent_node(self) -> IRNode | None:
        return self.parent

    @property
    def ops(self) -> BlockOps:
        """Returns a multi-pass Iterable of this block's operations."""
        return BlockOps(self)

    @property
    def ops_reverse(self) -> BlockReverseOps:
        """Returns a multi-pass Iterable of this block's operations."""
        return BlockReverseOps(self)

    def parent_op(self) -> Operation | None:
        return self.parent.parent if self.parent else None

    def parent_region(self) -> Region | None:
        return self.parent

    def parent_block(self) -> Block | None:
        return self.parent.parent.parent if self.parent and self.parent.parent else None

    def __repr__(self) -> str:
        return f"Block(_args={repr(self._args)}, num_ops={len(self.ops)})"

    @property
    def args(self) -> tuple[BlockArgument, ...]:
        """Returns the block arguments."""
        return self._args

    class BlockCallback(Protocol):
        def __call__(self, *args: BlockArgument) -> list[Operation]:
            ...

    def insert_arg(self, arg_type: Attribute, index: int) -> BlockArgument:
        """
        Insert a new argument with a given type to the arguments list at a specific index.
        Returns the new argument.
        """
        if index < 0 or index > len(self._args):
            raise Exception("Unexpected index")
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
            raise Exception("Attempting to delete an argument of the wrong block")
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
            raise Exception("Operation is not a children of the block.")
        for idx, block_op in enumerate(self.ops):
            if block_op is op:
                return idx
        assert False, "Unexpected xdsl error"

    def detach_op(self, op: Operation) -> Operation:
        """
        Detach an operation from the block.
        Returns the detached operation.
        """
        if op.parent is not self:
            raise Exception("Cannot detach operation from a different block.")
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
        Call a function on all operations contained in the block.
        If region_first is set, then the operation regions are iterated before the
        operation. If reverse is set, then the region, block, and operation lists are
        iterated in reverse order.
        """
        for op in self.ops_reverse if reverse else self.ops:
            yield from op.walk(reverse=reverse, region_first=region_first)

    @deprecated("Use walk(reverse=True) instead")
    def walk_reverse(self) -> Iterable[Operation]:
        """Call a function on all operations contained in the block in reverse order."""
        for op in self.ops_reverse:
            yield from op.walk_reverse()

    def verify(self) -> None:
        for operation in self.ops:
            if operation.parent != self:
                raise Exception(
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
        for op in self.ops:
            op.drop_all_references()

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

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(init=False)
class Region(IRNode):
    """A region contains a CFG of blocks. Regions are contained in operations."""

    class DEFAULT:
        """
        A marker to be used as a default parameter to functions when a default
        single-block region should be constructed.
        """

    blocks: list[Block] = field(default_factory=list)
    """Blocks contained in the region. The first block is the entry block."""

    parent: Operation | None = field(default=None, repr=False)
    """Operation containing the region."""

    def __init__(self, blocks: Block | Iterable[Block] = ()):
        super().__init__()
        self.blocks = []
        if isinstance(blocks, Block):
            blocks = (blocks,)
        for block in blocks:
            self.add_block(block)

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

    def __repr__(self) -> str:
        return f"Region(num_blocks={len(self.blocks)})"

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
        if len(self.blocks) != 1:
            raise ValueError(
                "'block' property of Region class is only available "
                "for single-block regions."
            )
        return self.blocks[0]

    def _attach_block(self, block: Block) -> None:
        """Attach a block to the region, and check that it has no parents."""
        if block.parent:
            raise ValueError(
                "Can't add to a region a block already attached to a region."
            )
        if block.is_ancestor(self):
            raise ValueError("Can't add a block to a region contained in the block.")
        block.parent = self

    def add_block(self, block: Block) -> None:
        """Add a block to the region."""
        self._attach_block(block)
        self.blocks.append(block)

    def insert_block(self, blocks: Block | list[Block], index: int) -> None:
        """
        Insert one or multiple blocks at a given index in the region.
        The blocks should not be attached to another region.
        """
        if index < 0 or index > len(self.blocks):
            raise ValueError(
                f"Can't insert block in index {index} in a block with "
                f"{len(self.blocks)} blocks."
            )
        if not isinstance(blocks, list):
            blocks = [blocks]
        for block in blocks:
            self._attach_block(block)
        self.blocks = self.blocks[:index] + blocks + self.blocks[index:]

    def get_block_index(self, block: Block) -> int:
        """Get the block position in a region."""
        if block.parent is not self:
            raise Exception("Block is not a child of the region.")
        for idx, region_block in enumerate(self.blocks):
            if region_block is block:
                return idx
        assert False, "Unexpected xdsl error"

    def detach_block(self, block: int | Block) -> Block:
        """
        Detach a block from the region.
        Returns the detached block.
        """
        if isinstance(block, Block):
            block_idx = self.get_block_index(block)
        else:
            block_idx = block
            block = self.blocks[block_idx]
        block.parent = None
        self.blocks = self.blocks[:block_idx] + self.blocks[block_idx + 1 :]
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
                value_mapper[block_arg] = new_block.args[idx]
            for op in block.ops:
                new_block.add_op(op.clone(value_mapper, block_mapper))

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

    @deprecated("Use walk(reverse=True) instead")
    def walk_reverse(self) -> Iterator[Operation]:
        """Call a function on all operations contained in the region in reverse order."""
        for block in reversed(self.blocks):
            yield from block.walk_reverse()

    def verify(self) -> None:
        for block in self.blocks:
            block.verify()
            if block.parent != self:
                raise Exception(
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
        region.blocks = self.blocks
        self.blocks = []
        for block in region.blocks:
            block.parent = region

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
