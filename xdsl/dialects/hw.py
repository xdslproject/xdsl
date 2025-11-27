"""
This is a stub of CIRCT’s hw dialect.
Follows definitions as of CIRCT commit `f8c7faec1e8447521a1ea9a0836b6923a132c79e`.

See [rationale](https://circt.llvm.org/docs/RationaleSymbols/) for symbols.
See external [documentation](https://circt.llvm.org/docs/Dialects/HW/).
"""

import abc
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import ClassVar, NamedTuple, cast, overload

from xdsl.dialects.builtin import (
    AnySignlessIntegerType,
    ArrayAttr,
    FlatSymbolRefAttr,
    FlatSymbolRefAttrConstr,
    IntAttr,
    IntegerType,
    LocationAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    OpaqueSyntaxAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AtLeast,
    IRDLOperation,
    RangeOf,
    VarConstraint,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, BaseParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    OpTrait,
    SingleBlockImplicitTerminator,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@dataclass
class InnerSymTarget:
    """The target of an inner symbol, the entity the symbol is a handle for.

    A None operation defines an invalid target, which is returned from InnerSymbolTable.lookup()
    when no matching operation is found. An invalid target is falsey when constrained to bool.
    """

    op: Operation | None = None
    field_id: int = 0
    port_idx: int | None = None

    def __bool__(self):
        return self.op is not None

    def is_port(self) -> bool:
        return self.port_idx is not None

    def is_field(self) -> bool:
        return self.field_id != 0

    def is_op_only(self) -> bool:
        return not self.is_field() and not self.is_port()

    @classmethod
    def get_target_for_subfield(
        cls, base: "InnerSymTarget", field_id: int
    ) -> "InnerSymTarget":
        """
        Return a target to the specified field within the given base.
        `field_id` is relative to the specified base target.
        """
        return cls(base.op, base.field_id + field_id, base.port_idx)


@irdl_attr_definition
class InnerRefAttr(ParametrizedAttribute):
    """This works like a symbol reference, but to a name inside a module."""

    name = "hw.innerNameRef"
    module_ref: FlatSymbolRefAttr
    # NB. upstream defines as “name” which clashes with Attribute.name
    sym_name: StringAttr

    def __init__(self, module: str | StringAttr, name: str | StringAttr) -> None:
        if isinstance(module, str):
            module = StringAttr(module)
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(SymbolRefAttr(module), name)

    @classmethod
    def get_from_operation(
        cls, op: Operation, sym_name: StringAttr, module_name: StringAttr
    ) -> "InnerRefAttr":
        """Get the InnerRefAttr for an operation and add the sym on it."""
        # NB: declared upstream, but no implementation to be found
        raise NotImplementedError

    def get_module(self) -> StringAttr:
        """Return the name of the referenced module."""
        return self.module_ref.root_reference

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<")
        symbol_ref = parser.parse_attribute()
        parser.parse_punctuation(">")
        if (
            not isinstance(symbol_ref, SymbolRefAttr)
            or len(symbol_ref.nested_references) != 1
        ):
            parser.raise_error("Expected a module and symbol reference")
        return [
            SymbolRefAttr(symbol_ref.root_reference),
            symbol_ref.nested_references.data[0],
        ]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_symbol_name(self.module_ref.root_reference.data)
            printer.print_string("::")
            printer.print_symbol_name(self.sym_name.data)


@dataclass(frozen=True)
class InnerSymbolTableTrait(OpTrait):
    """A trait for inner symbol table functionality on an operation."""

    def verify(self, op: Operation):
        # Insist that ops with InnerSymbolTable's provide a Symbol, this is
        # essential to how InnerRef's work.
        if not op.has_trait(trait := SymbolOpInterface):
            raise VerifyException(
                f"Operation {op.name} must have trait {trait.__name__}"
            )

        # InnerSymbolTable's must be directly nested within an InnerRefNamespaceTrait (or similar),
        # however don’t test InnerRefNamespace’s symbol lookups
        parent = op.parent_op()
        if parent is None or not parent.has_trait(trait := InnerRefNamespaceLike):
            raise VerifyException(
                f"Operation {op.name} with trait {type(self).__name__} must have a parent with trait {trait.__name__}"
            )


@dataclass
class InnerSymbolTable:
    """A class for lookups in inner symbol tables. Called InnerSymbolTable in upstream (name clash with trait)."""

    op: InitVar[Operation | None] = None
    symbol_table: dict[StringAttr, InnerSymTarget] = field(
        default_factory=dict[StringAttr, InnerSymTarget]
    )

    def __post_init__(self, op: Operation | None = None) -> None:
        pass
        # Here will populate self.symbol_table


@dataclass
class InnerSymbolTableCollection:
    """This class represents a collection of InnerSymbolTable."""

    symbol_tables: dict[Operation, InnerSymbolTable] = field(
        default_factory=dict[Operation, InnerSymbolTable], init=False
    )
    op: InitVar[Operation | None] = None

    def __post_init__(self, op: Operation | None = None) -> None:
        if op is None:
            return
        if not op.has_trait(trait := InnerRefNamespaceTrait):
            raise VerifyException(
                f"Operation {op.name} should have {trait.__name__} trait"
            )
        self.populate_and_verify_tables(op)

    def get_inner_symbol_table(self, op: Operation) -> InnerSymbolTable:
        """Returns the InnerSymolTable trait, ensuring `op` is in the collection"""
        if not op.has_trait(trait := InnerSymbolTableTrait):
            raise VerifyException(
                f"Operation {op.name} should have {trait.__name__} trait"
            )
        if op not in self.symbol_tables:
            self.symbol_tables[op] = InnerSymbolTable(op)
        return self.symbol_tables[op]

    def populate_and_verify_tables(self, inner_ref_ns_op: Operation):
        """Populate tables for all InnerSymbolTable operations in the given InnerRefNamespace operation, verifying each."""
        # Gather top-level operations that have the InnerSymbolTable trait.
        inner_sym_table_ops = (
            op for op in inner_ref_ns_op.walk() if op.has_trait(InnerSymbolTableTrait)
        )

        # Construct the tables
        for op in inner_sym_table_ops:
            if op in self.symbol_tables:
                raise VerifyException(
                    f"Trying to insert the same op twice in symbol tables: {op}"
                )
            self.symbol_tables[op] = InnerSymbolTable(op)


class InnerRefUserOpInterfaceTrait(OpTrait):
    """This interface describes an operation that may use a `InnerRef`. This
    interface allows for users of inner symbols to hook into verification and
    other inner symbol related utilities that are either costly or otherwise
    disallowed within a traditional operation."""

    def verify_inner_refs(self, op: Operation, namespace: "InnerRefNamespace"):
        """Verify the inner ref uses held by this operation."""
        ...


@dataclass(frozen=True)
class InnerRefNamespaceTrait(OpTrait):
    """Trait for operations defining a new scope for InnerRef's. Operations with this trait must be a SymbolTable."""

    def verify(self, op: Operation):
        if not op.has_trait(trait := SymbolTable):
            raise VerifyException(
                f"Operation {op.name} must have trait {trait.__name__}"
            )

        # Upstreams verifies that len(op.regions) == 1 and len(op.regions[0].blocks) == 1
        # however this is already checked as part of SymbolTable, so would be redundant to re-check here

        namespace = InnerRefNamespace(op)

        for inner_op in op.walk():
            inner_ref_user_op_trait = inner_op.get_trait(InnerRefUserOpInterfaceTrait)
            if inner_ref_user_op_trait is not None:
                inner_ref_user_op_trait.verify_inner_refs(inner_op, namespace)


class InnerRefNamespaceLike(abc.ABC, OpTrait):
    """Trait-metaclass to check whether an operation is explicitly an IRN or appears compatible."""


InnerRefNamespaceLike.register(SymbolTable)
InnerRefNamespaceLike.register(InnerRefNamespaceTrait)


@dataclass
class InnerRefNamespace:
    """Class to perform symbol lookups within a InnerRef namespace, used during verification.
    Combines InnerSymbolTableCollection with a SymbolTable for resolution of InnerRefAttrs.

    Inner symbols are more costly than normal symbols, with tricker verification. For this reason,
    verification is driven as a trait verifier on InnerRefNamespace which constructs and verifies InnerSymbolTables in parallel.
    See: https://circt.llvm.org/docs/RationaleSymbols/#innerrefnamespace
    """

    inner_sym_tables: InnerSymbolTableCollection = field(init=False)
    inner_ref_ns_op: InitVar[Operation]

    def __init__(self, inner_ref_ns_op: Operation):
        self.inner_sym_tables = InnerSymbolTableCollection(inner_ref_ns_op)


@irdl_attr_definition
class InnerSymPropertiesAttr(ParametrizedAttribute):
    name = "hw.innerSymProps"

    # NB. upstream defines as “name” which clashes with Attribute.name
    sym_name: StringAttr
    field_id: IntAttr
    sym_visibility: StringAttr

    def __init__(
        self,
        sym: str | StringAttr,
        field_id: int | IntAttr = 0,
        sym_visibility: str | StringAttr = "public",
    ) -> None:
        if isinstance(sym, str):
            sym = StringAttr(sym)
        if isinstance(field_id, int):
            field_id = IntAttr(field_id)
        if isinstance(sym_visibility, str):
            sym_visibility = StringAttr(sym_visibility)
        super().__init__(sym, field_id, sym_visibility)

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> tuple[StringAttr, IntAttr, StringAttr]:
        parser.parse_punctuation("<")
        sym_name = parser.parse_symbol_name()
        parser.parse_punctuation(",")
        field_id = parser.parse_integer(allow_negative=False, allow_boolean=False)
        parser.parse_punctuation(",")
        sym_visibility = parser.parse_identifier()
        if sym_visibility not in {"public", "private", "nested"}:
            parser.raise_error('Expected "public", "private", or "nested"')
        parser.parse_punctuation(">")
        return (sym_name, IntAttr(field_id), StringAttr(sym_visibility))

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<@")
        printer.print_string(self.sym_name.data)
        printer.print_string(",")
        printer.print_int(self.field_id.data)
        printer.print_string(",")
        printer.print_string(self.sym_visibility.data)
        printer.print_string(">")

    def verify(self):
        if not self.sym_name or not self.sym_name.data:
            raise VerifyException("inner symbol cannot have empty name")


@irdl_attr_definition
class InnerSymAttr(
    ParametrizedAttribute, Iterable[InnerSymPropertiesAttr], OpaqueSyntaxAttribute
):
    """Inner symbol definition

    Defines the properties of an inner_sym attribute. It specifies the symbol name and symbol
    visibility for each field ID. For any ground types, there are no subfields and the field ID is 0.
    For aggregate types, a unique field ID is assigned to each field by visiting them in a
    depth-first pre-order.

    The custom assembly format ensures that for ground types, only `@<sym_name>` is printed.
    """

    name = "hw.innerSym"

    props: ArrayAttr[InnerSymPropertiesAttr]

    @overload
    def __init__(self) -> None:
        # Create an empty array, represents an invalid InnerSym.
        ...

    @overload
    def __init__(self, syms: str | StringAttr) -> None: ...

    @overload
    def __init__(
        self, syms: Sequence[InnerSymPropertiesAttr] | ArrayAttr[InnerSymPropertiesAttr]
    ) -> None: ...

    def __init__(
        self,
        syms: (
            str
            | StringAttr
            | Sequence[InnerSymPropertiesAttr]
            | ArrayAttr[InnerSymPropertiesAttr]
        ) = [],
    ) -> None:
        if isinstance(syms, str | StringAttr):
            syms = [InnerSymPropertiesAttr(syms)]
        if not isinstance(syms, ArrayAttr):
            syms = ArrayAttr(syms)
        super().__init__(syms)

    def get_sym_if_exists(self, field_id: IntAttr | int) -> StringAttr | None:
        """Get the inner sym name for field_id, if it exists."""
        if not isinstance(field_id, IntAttr):
            field_id = IntAttr(field_id)

        for prop in self.props:
            if field_id == prop.field_id:
                return prop.sym_name

    def get_sym_name(self) -> StringAttr | None:
        """Get the inner sym name for field_id=0, if it exists."""
        return self.get_sym_if_exists(0)

    def __len__(self) -> int:
        """Get the number of inner symbols defined."""
        return len(self.props)

    def __iter__(self) -> Iterator[InnerSymPropertiesAttr]:
        """Iterator for all the InnerSymPropertiesAttr."""
        return iter(self.props)

    def erase(self, field_id: IntAttr | int) -> "InnerSymAttr":
        """Return an InnerSymAttr with the inner symbol for the specified field_id removed."""
        if not isinstance(field_id, IntAttr):
            field_id = IntAttr(field_id)
        return InnerSymAttr([prop for prop in self.props if prop.field_id != field_id])

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> list[ArrayAttr[InnerSymPropertiesAttr]]:
        if (sym_name := parser.parse_optional_symbol_name()) is not None:
            return [ArrayAttr([InnerSymPropertiesAttr(sym_name, 0, "public")])]

        data = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            lambda: InnerSymPropertiesAttr.parse_parameters(parser),
        )
        return [ArrayAttr(InnerSymPropertiesAttr(*tup) for tup in data)]

    def print_parameters(self, printer: Printer):
        if (
            len(self) == 1
            and (sym_name := self.get_sym_name()) is not None
            and self.props.data[0].sym_visibility.data == "public"
            and self.props.data[0].field_id.data == 0
        ):
            printer.print_string("@")
            printer.print_string(sym_name.data)
        else:
            printer.print_string("[")
            printer.print_list(
                sorted(self.props, key=lambda prop: prop.field_id.data),
                lambda prop: prop.print_parameters(printer),
            )
            printer.print_string("]")


class Direction(Enum):
    """
    Represents the direction of a module port.
    """

    # TODO: support INOUT direction (https://github.com/xdslproject/xdsl/issues/2368)
    INPUT = 0
    OUTPUT = 1

    @staticmethod
    def parse_optional(parser: BaseParser, short: bool = False) -> "Direction | None":
        if parser.parse_optional_keyword("input" if not short else "in"):
            return Direction.INPUT
        elif parser.parse_optional_keyword("output" if not short else "out"):
            return Direction.OUTPUT
        else:
            return None

    @staticmethod
    def parse(parser: BaseParser, short: bool = False) -> "Direction":
        if (direction := Direction.parse_optional(parser, short)) is None:
            return parser.raise_error("invalid port direction")
        return direction

    def print(self, printer: Printer, short: bool = False) -> None:
        match self:
            case Direction.INPUT:
                printer.print_string("input" if not short else "in")
            case Direction.OUTPUT:
                printer.print_string("output" if not short else "out")

    def is_input_like(self) -> bool:
        return self == Direction.INPUT

    def is_output_like(self) -> bool:
        return self == Direction.OUTPUT


@irdl_attr_definition
class DirectionAttr(Data[Direction]):
    """
    Represents a ModulePort's direction. This attribute does not
    exist in CIRCT but is useful in xDSL to give structure to ModuleType.
    """

    name = "hw.direction"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> Direction:
        with parser.in_angle_brackets():
            return Direction.parse(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.data.print(printer)


@irdl_attr_definition
class ModulePort(ParametrizedAttribute):
    """
    Represents a ModulePort. This attribute does not exist in CIRCT
    but is useful in xDSL to give structure to ModuleType.
    """

    name = "hw.modport"

    port_name: StringAttr
    type: TypeAttribute
    dir: DirectionAttr


@irdl_attr_definition
class ModuleType(ParametrizedAttribute, TypeAttribute):
    name = "hw.modty"

    ports: ArrayAttr[ModulePort]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        def parse_port() -> ModulePort:
            direction = Direction.parse(parser)
            name = (
                parser.parse_optional_identifier()
                or parser.parse_optional_str_literal()
            )
            if name is None:
                parser.raise_error("expected port name as identifier or string literal")

            parser.parse_punctuation(":")
            typ = parser.parse_type()

            return ModulePort(StringAttr(name), typ, DirectionAttr(direction))

        return [
            ArrayAttr(
                parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parse_port)
            )
        ]

    def print_parameters(self, printer: Printer):
        def print_port(port: ModulePort):
            port.dir.data.print(printer)
            printer.print_string(" ")
            printer.print_identifier_or_string_literal(port.port_name.data)
            printer.print_string(" : ")
            printer.print_attribute(port.type)

        with printer.in_angle_brackets():
            printer.print_list(self.ports.data, print_port)


@irdl_attr_definition
class ParamDeclAttr(ParametrizedAttribute):
    name = "hw.param.decl"

    port_name: StringAttr
    type: TypeAttribute

    @classmethod
    def parse_free_standing_parameters(
        cls, parser: AttrParser, only_accept_string_literal_name: bool = False
    ) -> tuple[StringAttr, TypeAttribute]:
        """
        Parses the parameter declaration without the encompassing angle brackets.
        If only_accept_string_literal_name is True, the parser will not accept
        the name of the parameter to be an identifier but only as a string literal.
        """

        name = parser.parse_optional_str_literal()
        if name is None:
            if only_accept_string_literal_name:
                parser.raise_error("expected parameter name as string literal")
            name = parser.expect(
                parser.parse_optional_identifier, "expected parameter name"
            )
        parser.parse_punctuation(":")
        typ = parser.parse_attribute()
        if not isinstance(typ, TypeAttribute):
            parser.raise_error("expected type attribute for parameter")

        if parser.parse_optional_punctuation("=") is not None:
            # TODO: support default values for parameters (https://github.com/xdslproject/xdsl/issues/2367)
            parser.raise_error("default values for parameters are not yet supported")

        return (StringAttr(name), typ)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            return cls.parse_free_standing_parameters(
                parser, only_accept_string_literal_name=True
            )

    def print_free_standing_parameters(
        self, printer: Printer, print_name_as_string_literal: bool = False
    ):
        """
        Prints the parameter declaration without the encompassing angle brackets.
        If print_name_as_string_literal is True, the name of the parameter will
        never be printed as an identifier but only as a string literal.
        """
        if print_name_as_string_literal:
            printer.print_attribute(self.port_name)
        else:
            printer.print_identifier_or_string_literal(self.port_name.data)
        printer.print_string(": ")
        printer.print_attribute(self.type)

    def print_parameters(self, printer: Printer):
        with printer.in_angle_brackets():
            self.print_free_standing_parameters(
                printer, print_name_as_string_literal=True
            )


@irdl_attr_definition
class ArrayType(ParametrizedAttribute, TypeAttribute):
    """
    Fixed-sized array
    """

    name = "hw.array"

    element_type: IntegerType
    size_attr: IntAttr

    def __init__(
        self,
        element_type: AnySignlessIntegerType,
        size_attr: IntAttr | int,
    ):
        if isinstance(size_attr, int):
            size_attr = IntAttr(size_attr)
        super().__init__(
            element_type,
            size_attr,
        )

    def __len__(self) -> int:
        return self.size_attr.data

    def get_element_type(self) -> IntegerType:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<", " in hw.array type")
        size_attr, type = parser.parse_ranked_shape()
        if len(size_attr) != 1:
            parser.raise_error("Expected one size in hw.array type")
        size_attr = IntAttr(size_attr[0])
        parser.parse_punctuation(">", " in hw.array type")
        return [type, size_attr]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(len(self))
            printer.print_string("x")
            printer.print_attribute(self.get_element_type())


class HWModuleLike(OpTrait, abc.ABC):
    """
    Represents an operation that can be interpreted as a hardware module.
    """

    @classmethod
    @abc.abstractmethod
    def get_hw_module_type(cls, op: Operation) -> ModuleType:
        """
        Returns the type of the module.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def set_hw_module_type(cls, op: Operation, module_type: ModuleType) -> None:
        """
        Sets the type of the module.
        """
        raise NotImplementedError()


class ParsedModuleHeader(NamedTuple):
    """
    Represents the parsed common base of all modules.
    It consists mostly of the ports and parameters of the
    module.
    """

    class ModuleArg(NamedTuple):
        port_dir: Direction
        port_name: str
        port_ssa: Parser.Argument | None
        port_type: TypeAttribute
        port_attrs: dict[str, Attribute]
        port_location: LocationAttr | None

    visibility: StringAttr | None
    module_name: StringAttr
    parameters: ArrayAttr[ParamDeclAttr]
    args: list[ModuleArg]

    def get_module_type(self) -> ModuleType:
        return ModuleType(
            ArrayAttr(
                tuple(
                    ModulePort(
                        StringAttr(arg.port_name),
                        arg.port_type,
                        DirectionAttr(arg.port_dir),
                    )
                    for arg in self.args
                )
            )
        )

    @classmethod
    def parse(cls, parser: Parser) -> "ParsedModuleHeader":
        def parse_optional_port_name() -> str | None:
            return (
                parser.parse_optional_identifier()
                or parser.parse_optional_str_literal()
            )

        def parse_module_arg() -> ParsedModuleHeader.ModuleArg:
            port_dir = Direction.parse(parser, short=True)
            if port_dir.is_input_like():
                port_ssa = parser.parse_argument(expect_type=False)
                port_name = parse_optional_port_name()
                if port_name is None:
                    port_name = port_ssa.name.text[1:]
            else:
                port_ssa = None
                port_name = parse_optional_port_name()
                if port_name is None:
                    parser.raise_error(
                        "expected identifier or string literal as port name"
                    )
            parser.parse_punctuation(":")
            port_type = parser.parse_attribute()
            if not isinstance(port_type, TypeAttribute):
                parser.raise_error("port type must be a type attribute")

            port_attrs = parser.parse_optional_attr_dict()
            port_location = parser.parse_optional_location()

            # Resolve the argument
            if port_ssa is not None:
                port_ssa = Parser.Argument(port_ssa.name, port_type)

            return ParsedModuleHeader.ModuleArg(
                port_dir, port_name, port_ssa, port_type, port_attrs, port_location
            )

        sym_visibility = parser.parse_optional_visibility_keyword()
        name = parser.parse_symbol_name()
        parameters = parser.parse_optional_comma_separated_list(
            parser.Delimiter.ANGLE,
            lambda: ParamDeclAttr(
                *ParamDeclAttr.parse_free_standing_parameters(parser)
            ),
        )
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parse_module_arg
        )

        return cls(
            visibility=sym_visibility,
            module_name=name,
            parameters=ArrayAttr(parameters if parameters is not None else []),
            args=args,
        )


def print_module_header(
    *,
    printer: Printer,
    visibility: StringAttr | None,
    module_name: StringAttr,
    parameters: ArrayAttr[ParamDeclAttr] | None,
    arg_ssa_iter: Iterable[SSAValue | str],
    module_type: ModuleType,
):
    """
    Prints the common header of a module.
    The `arg_ssa_iter` parameter provides the SSA names to be
    used for input-like parameters, either using the print
    name of an SSA value, or a string decided beforehand.
    """
    if visibility is not None:
        printer.print_string(" ")
        printer.print_string(visibility.data)
    printer.print_string(" ")
    printer.print_symbol_name(module_name.data)

    # Print parameters
    if parameters is not None and len(parameters.data) != 0:
        with printer.in_angle_brackets():
            printer.print_list(
                parameters.data,
                lambda x: x.print_free_standing_parameters(printer),
            )

    arg_iter = iter(arg_ssa_iter)

    def print_port(port: ModulePort):
        ssa_arg = next(arg_iter) if port.dir.data.is_input_like() else None
        port.dir.data.print(printer, short=True)
        printer.print_string(" ")

        # Print argument
        if ssa_arg is not None:
            if isinstance(ssa_arg, SSAValue):
                used_name = printer.print_ssa_value(ssa_arg)
            else:
                assert isinstance(ssa_arg, str)
                printer.print_string("%")
                printer.print_string(ssa_arg)
                used_name = ssa_arg
            if port.port_name.data != used_name:
                printer.print_string(" ")
                printer.print_identifier_or_string_literal(port.port_name.data)
        else:
            printer.print_identifier_or_string_literal(port.port_name.data)
        printer.print_string(": ")
        printer.print_attribute(port.type)

    with printer.in_parens():
        printer.print_list(module_type.ports.data, print_port)


_MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT: list[str] = [
    "sym_name",
    "module_type",
    "parameters",
    "sym_visibility",
]


class HWModulesHWModuleLike(HWModuleLike):
    @classmethod
    def get_hw_module_type(cls, op: Operation) -> ModuleType:
        assert isinstance(op, HWModuleOp | HWModuleExternOp)
        return op.module_type

    @classmethod
    def set_hw_module_type(cls, op: Operation, module_type: ModuleType) -> None:
        assert isinstance(op, HWModuleOp | HWModuleExternOp)
        op.module_type = module_type


@irdl_op_definition
class HWModuleOp(IRDLOperation):
    """
    Represents a Verilog module, including a given name, a list of ports,
    a list of parameters, and a body that represents the connections within
    the module.
    """

    name = "hw.module"

    sym_name = attr_def(SymbolNameConstraint())
    module_type = attr_def(ModuleType)
    sym_visibility = opt_attr_def(StringAttr)
    parameters = opt_attr_def(ArrayAttr[ParamDeclAttr])

    body = region_def("single_block")

    traits = lazy_traits_def(
        lambda: (
            SymbolOpInterface(),
            IsolatedFromAbove(),
            SingleBlockImplicitTerminator(OutputOp),
            HWModulesHWModuleLike(),
        )
    )

    def __init__(
        self,
        sym_name: StringAttr,
        module_type: ModuleType,
        body: Region,
        parameters: ArrayAttr[ParamDeclAttr] = ArrayAttr([]),
        visibility: str | StringAttr | None = None,
    ):
        attributes: dict[str, Attribute] = {
            "sym_name": sym_name,
            "module_type": module_type,
            "parameters": parameters,
        }

        if visibility:
            if isinstance(visibility, str):
                visibility = StringAttr(visibility)
            attributes["sym_visibility"] = visibility

        return super().__init__(
            attributes=attributes,
            regions=[body],
        )

    def verify_(self) -> None:
        if self.parameters is not None:
            # FIXME: once xDSL supports typed attributes, check that parameter
            # types are consistent with their default values
            param_names = [param.port_name.data for param in self.parameters.data]
            if len(set(param_names)) != len(param_names):
                raise VerifyException("module has two parameters of same name")
        block_args = iter(self.body.block.args)
        for i, port in enumerate(self.module_type.ports.data):
            if not port.dir.data.is_input_like():
                continue
            if (next_block_arg := next(block_args, None)) is None:
                raise VerifyException("missing block arguments in module block")
            if port.type != next_block_arg.type:
                raise VerifyException(
                    f"input-like port #{i} has inconsistent type with its matching "
                    f"module block argument (expected {port.type}, block argument "
                    f"is of type {next_block_arg.type})"
                )
        if next(block_args, None) is not None:
            raise VerifyException("too many block arguments in module block")

    @classmethod
    def parse(cls, parser: Parser) -> "HWModuleOp":
        module_header = ParsedModuleHeader.parse(parser)

        attrs = parser.parse_optional_attr_dict_with_keyword(
            _MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT
        )

        # Create a body region suitable for the ports of the module.
        region_args = tuple(arg.port_ssa for arg in module_header.args if arg.port_ssa)
        body = parser.parse_region(region_args)

        module_op = cls(
            module_header.module_name,
            module_header.get_module_type(),
            body,
            module_header.parameters,
            module_header.visibility,
        )

        if attrs is not None:
            module_op.attributes.update(attrs.data)

        return module_op

    def print(self, printer: Printer):
        print_module_header(
            printer=printer,
            visibility=self.sym_visibility,
            module_name=self.sym_name,
            parameters=self.parameters,
            arg_ssa_iter=self.body.block.args,
            module_type=self.module_type,
        )
        printer.print_op_attributes(
            self.attributes,
            reserved_attr_names=_MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT,
            print_keyword=True,
        )
        printer.print_string(" ")
        printer.print_region(self.body, print_entry_block_args=False)


@irdl_op_definition
class HWModuleExternOp(IRDLOperation):
    """
    The "hw.module.extern" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.

    The 'verilogName' attribute (when present) specifies the spelling of the
    module name in Verilog we can use.
    """

    name = "hw.module.extern"

    sym_name = attr_def(SymbolNameConstraint())
    module_type = attr_def(ModuleType)
    sym_visibility = opt_attr_def(StringAttr)
    parameters = opt_attr_def(ArrayAttr[ParamDeclAttr])
    verilog_name = opt_attr_def(StringAttr, attr_name="verilogName")

    traits = lazy_traits_def(
        lambda: (
            SymbolOpInterface(),
            HWModulesHWModuleLike(),
        )
    )

    def __init__(
        self,
        sym_name: StringAttr,
        module_type: ModuleType,
        parameters: ArrayAttr[ParamDeclAttr] = ArrayAttr([]),
        visibility: str | StringAttr | None = None,
    ):
        attributes: dict[str, Attribute] = {
            "sym_name": sym_name,
            "module_type": module_type,
            "parameters": parameters,
        }

        if visibility:
            if isinstance(visibility, str):
                visibility = StringAttr(visibility)
            attributes["sym_visibility"] = visibility

        return super().__init__(attributes=attributes)

    @classmethod
    def parse(cls, parser: Parser) -> "HWModuleExternOp":
        module_header = ParsedModuleHeader.parse(parser)

        attrs = parser.parse_optional_attr_dict_with_keyword(
            _MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT
        )

        module_op = cls(
            module_header.module_name,
            module_header.get_module_type(),
            module_header.parameters,
            module_header.visibility,
        )

        if attrs is not None:
            module_op.attributes.update(attrs.data)

        return module_op

    def print(self, printer: Printer):
        # TODO: use the actual port name when it is a valid SSA name
        arg_ssa_names = tuple(
            f"port{i}"
            for i, port in enumerate(self.module_type.ports.data)
            if port.dir.data.is_input_like()
        )
        print_module_header(
            printer=printer,
            visibility=self.sym_visibility,
            module_name=self.sym_name,
            parameters=self.parameters,
            arg_ssa_iter=arg_ssa_names,
            module_type=self.module_type,
        )
        printer.print_op_attributes(
            self.attributes,
            reserved_attr_names=_MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT,
            print_keyword=True,
        )


@irdl_op_definition
class InstanceOp(IRDLOperation):
    """
    This represents an instance of a module. The inputs and outputs are the
    referenced module's inputs and outputs. The argNames and resultNames
    attributes must match the referenced module's input and output names.
    """

    name = "hw.instance"

    instance_name = attr_def(StringAttr, attr_name="instanceName")
    module_name = attr_def(FlatSymbolRefAttrConstr, attr_name="moduleName")
    inputs = var_operand_def()
    outputs = var_result_def()
    arg_names = attr_def(ArrayAttr[StringAttr], attr_name="argNames")
    result_names = attr_def(ArrayAttr[StringAttr], attr_name="resultNames")
    inner_sym = opt_attr_def(InnerSymAttr)

    def __init__(
        self,
        instance_name: str,
        module_name: FlatSymbolRefAttr,
        inputs: Iterable[tuple[str, SSAValue]],
        outputs: Iterable[tuple[str, TypeAttribute]],
        inner_sym: InnerSymAttr | None = None,
    ):
        arg_names = ArrayAttr(StringAttr(port[0]) for port in inputs)
        result_names = ArrayAttr(StringAttr(port[0]) for port in outputs)
        attributes: dict[str, Attribute] = {
            "instanceName": StringAttr(instance_name),
            "moduleName": module_name,
            "argNames": arg_names,
            "resultNames": result_names,
        }
        if inner_sym is not None:
            attributes["inner_sym"] = inner_sym
        super().__init__(
            operands=(tuple(port[1] for port in inputs),),
            result_types=(tuple(port[1] for port in outputs),),
            attributes=attributes,
        )

    def verify_(self) -> None:
        if len(self.arg_names.data) != len(self.inputs):
            raise VerifyException(
                "Instance has a different amount of argument names "
                f"({len(self.arg_names.data)}) "
                f"and arguments ({len(self.inputs)})"
            )
        if len(self.result_names.data) != len(self.outputs):
            raise VerifyException(
                "Instance has a different amount of result names "
                f"({len(self.result_names.data)}) "
                f"and results ({len(self.outputs)})"
            )

        module = SymbolTable.lookup_symbol(self, self.module_name)
        if module is None:
            raise VerifyException(f"Module {self.module_name} not found")

        hw_module_like = module.get_trait(HWModuleLike)
        if hw_module_like is None:
            raise VerifyException(
                f"Module {self.module_name} must be a HWModuleLike, "
                f"found '{module.name}'"
            )

        def check_same_or_exception(
            reference: Iterable[str], candidate: Iterable[str], kind: str
        ):
            reference_set = set(reference)
            visited: set[str] = set()
            for candidate in candidate:
                if candidate in visited:
                    raise VerifyException(
                        f"Multiple definitions for {kind} '{candidate}'"
                    )
                visited.add(candidate)
                if candidate not in reference_set:
                    raise VerifyException(f"Unknown {kind} '{candidate}'")
                reference_set.remove(candidate)
            if len(reference_set) != 0:
                raise VerifyException(f"Missing {kind} '{reference_set.pop()}'")

        module_args = (
            port.port_name.data
            for port in hw_module_like.get_hw_module_type(module).ports
            if port.dir.data.is_input_like()
        )
        result_args = (
            port.port_name.data
            for port in hw_module_like.get_hw_module_type(module).ports
            if port.dir.data.is_output_like()
        )

        check_same_or_exception(
            module_args, (arg.data for arg in self.arg_names.data), "input port"
        )

        check_same_or_exception(
            result_args,
            (result.data for result in self.result_names.data),
            "output port",
        )

    @classmethod
    def parse(cls, parser: Parser) -> "InstanceOp":
        instance_name = parser.parse_str_literal(" (instance name)")
        inner_sym = None
        if parser.parse_optional_keyword("sym") is not None:
            inner_sym = parser.parse_attribute()
            if not isinstance(inner_sym, InnerSymAttr):
                parser.raise_error("Expected inner symbol attribute")
        module_name = parser.parse_attribute()
        if (
            not isinstance(module_name, SymbolRefAttr)
            or len(module_name.nested_references.data) != 0
        ):
            parser.raise_error("Expected flat symbol reference")
        if parser.parse_optional_punctuation("<") is not None:
            parser.raise_error("Instance parameters are not supported yet")

        def parse_input_port():
            port_name = (
                parser.parse_optional_str_literal()
                or parser.parse_optional_identifier()
            )
            if port_name is None:
                parser.raise_error("Expected port name as identifier or string literal")
            parser.parse_punctuation(":")
            port_operand = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            port_type = parser.parse_type()
            port_value = parser.resolve_operand(port_operand, port_type)
            return (port_name, port_value)

        def parse_output_port():
            port_name = (
                parser.parse_optional_str_literal()
                or parser.parse_optional_identifier()
            )
            if port_name is None:
                parser.raise_error("Expected port name as identifier or string literal")
            parser.parse_punctuation(":")
            port_type = parser.parse_type()
            return (port_name, port_type)

        input_ports = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parse_input_port, "input port list expected"
        )
        parser.parse_punctuation("->")
        output_ports = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parse_output_port, "output port list expected"
        )
        attributes_attr = parser.parse_optional_attr_dict_with_reserved_attr_names(
            (
                "instanceName",
                "moduleName",
                "argNames",
                "resultNames",
                "inner_sym",
            )
        )
        attributes = dict(attributes_attr.data) if attributes_attr is not None else {}

        operands = tuple(port[1] for port in input_ports)
        result_types = tuple(port[1] for port in output_ports)
        attributes["instanceName"] = StringAttr(instance_name)
        attributes["moduleName"] = module_name
        attributes["argNames"] = ArrayAttr(StringAttr(port[0]) for port in input_ports)
        attributes["resultNames"] = ArrayAttr(
            StringAttr(port[0]) for port in output_ports
        )
        if inner_sym is not None:
            attributes["inner_sym"] = inner_sym
        return cls.create(
            operands=operands, result_types=result_types, attributes=attributes
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.instance_name)
        printer.print_string(" ")
        if self.inner_sym is not None:
            printer.print_string("sym ")
            printer.print_attribute(self.inner_sym)
            printer.print_string(" ")
        printer.print_attribute(self.module_name)

        def print_input_port(name: str, operand: SSAValue):
            printer.print_identifier_or_string_literal(name)
            printer.print_string(": ")
            printer.print_operand(operand)
            printer.print_string(": ")
            printer.print_attribute(operand.type)

        def print_output_port(name: str, port_type: Attribute):
            printer.print_identifier_or_string_literal(name)
            printer.print_string(": ")
            printer.print_attribute(port_type)

        with printer.in_parens():
            printer.print_list(
                zip((name.data for name in self.arg_names), self.operands),
                lambda x: print_input_port(*x),
            )
        printer.print_string(" -> ")
        with printer.in_parens():
            printer.print_list(
                zip(
                    (name.data for name in self.result_names),
                    self.result_types,
                ),
                lambda x: print_output_port(*x),
            )
        printer.print_op_attributes(
            self.attributes,
            reserved_attr_names=(
                "instanceName",
                "moduleName",
                "argNames",
                "resultNames",
                "inner_sym",
            ),
        )


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "hw.output"

    inputs = var_operand_def()

    traits = traits_def(IsTerminator(), HasParent(HWModuleOp))

    def __init__(self, ops: Sequence[SSAValue | Operation]):
        super().__init__(operands=[ops])

    def verify_(self) -> None:
        parent = self.parent_op()
        assert isinstance(parent, HWModuleOp)

        expected_results = tuple(
            port.type
            for port in parent.module_type.ports.data
            if port.dir.data == Direction.OUTPUT
        )

        if len(expected_results) != len(self.inputs):
            raise VerifyException(
                f"wrong amount of output values (expected {len(expected_results)}, got {len(self.inputs)})"
            )

        for i, (got, expected) in enumerate(zip(self.inputs, expected_results)):
            if got.type != expected:
                raise VerifyException(
                    f"output #{i} is of unexpected type (expected {expected}, got {got.type})"
                )

    @classmethod
    def parse(cls, parser: Parser) -> "OutputOp":
        operands = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )
        if operands is None:
            return cls(())

        parser.parse_punctuation(":")
        types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_attribute
        )
        operands = parser.resolve_operands(operands, types, parser.pos)
        return cls(operands)

    def print(self, printer: Printer):
        if len(self.inputs) == 0:
            return

        printer.print_string(" ")
        printer.print_list(self.inputs, printer.print_operand)
        printer.print_string(" : ")
        printer.print_list(self.inputs.types, printer.print_attribute)


@irdl_op_definition
class ArrayCreateOp(IRDLOperation):
    """
    Create an array from values.
    Creates an array from a variable set of values. One or more values must be listed.
    """

    I: ClassVar = VarConstraint("I", AnyAttr())  # Constrain all types to be equal

    name = "hw.array_create"
    inputs = var_operand_def(RangeOf(I).of_length(AtLeast(1)))
    result = result_def(ArrayType)

    def __init__(
        self, first_input: Operation | SSAValue, *other_inputs: Operation | SSAValue
    ):
        inputs = (first_input, *other_inputs)
        el_type = SSAValue.get(first_input).type
        assert isa(el_type, AnySignlessIntegerType)
        out_type = ArrayType(el_type, len(inputs))
        super().__init__(operands=(inputs,), result_types=[out_type])

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_list(self.inputs, printer.print_operand)
        printer.print_string(" : ")
        printer.print_attribute(self.inputs[0].type)

    @classmethod
    def parse(cls, parser: Parser) -> "ArrayCreateOp":
        operands = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        in_type = parser.parse_type()
        types = [in_type for _ in range(len(operands))]
        operands = parser.resolve_operands(operands, types, parser.pos)
        return cls(*operands)


@irdl_op_definition
class ArrayGetOp(IRDLOperation):
    """
    Extract an element from an array
    Extracts the element at index from the given input array.
    The index must be exactly ceil(log2(length(input))) bits wide.
    """

    name = "hw.array_get"
    input = operand_def(ArrayType)
    index = operand_def(IntegerType)
    result = result_def(IntegerType)

    def __init__(self, input: Operation | SSAValue, index: Operation | SSAValue):
        typ = SSAValue.get(input).type
        assert isinstance(typ, ArrayType)
        out_type = typ.get_element_type()
        super().__init__(operands=[input, index], result_types=[out_type])

    def verify_(self) -> None:
        input_typ = cast(ArrayType, self.input.type)  # Checked by IRDL
        index_typ = cast(IntegerType, self.index.type)  # Checked by IRDL
        index_width = index_typ.bitwidth
        shape_width = (len(input_typ) - 1).bit_length()
        if index_width != shape_width:
            raise VerifyException(
                f"The index ({index_width} bits wide) must be exactly ceil(log2(length(input))) = {shape_width} bits wide"
            )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.input)
        with printer.in_square_brackets():
            printer.print_operand(self.index)
        printer.print_string(" : ")
        printer.print_attribute(self.input.type)
        printer.print_string(", ")
        printer.print_attribute(self.index.type)

    @classmethod
    def parse(cls, parser: Parser) -> "ArrayGetOp":
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        index = parser.parse_unresolved_operand()
        parser.parse_punctuation("]")
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_punctuation(",")
        index_type = parser.parse_type()
        operands = parser.resolve_operands(
            [input, index], [input_type, index_type], parser.pos
        )
        return cls(operands[0], operands[1])


HW = Dialect(
    "hw",
    [
        ArrayCreateOp,
        ArrayGetOp,
        HWModuleExternOp,
        HWModuleOp,
        InstanceOp,
        OutputOp,
    ],
    [
        ArrayType,
        DirectionAttr,
        InnerRefAttr,
        InnerSymPropertiesAttr,
        InnerSymAttr,
        ModulePort,
        ModuleType,
        ParamDeclAttr,
    ],
)
