"""
This is a stub of CIRCT’s hw dialect.
Up to date as of commit `TODO`

[1] https://circt.llvm.org/docs/Dialects/HW/
[2] https://circt.llvm.org/docs/RationaleSymbols/
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import overload

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FlatSymbolRefAttr,
    IntAttr,
    LocationAttr,
    NoneAttr,
    ParameterDef,
    StringAttr,
    SymbolNameAttr,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
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
    IRDLOperation,
    SingleBlockRegion,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    region_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, BaseParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsTerminator,
    OpTrait,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException


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
    module_ref: ParameterDef[FlatSymbolRefAttr]
    # NB. upstream defines as “name” which clashes with Attribute.name
    sym_name: ParameterDef[StringAttr]

    def __init__(self, module: str | StringAttr, name: str | StringAttr) -> None:
        if isinstance(module, str):
            module = StringAttr(module)
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__([FlatSymbolRefAttr(module), name])

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
            FlatSymbolRefAttr(symbol_ref.root_reference),
            symbol_ref.nested_references.data[0],
        ]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<@")
        printer.print_string(self.module_ref.root_reference.data)
        printer.print_string("::@")
        printer.print_string(self.sym_name.data)
        printer.print_string(">")


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

        # InnerSymbolTable's must be directly nested within an InnerRefNamespaceTrait,
        # however don’t test InnerRefNamespace’s symbol lookups
        parent = op.parent_op()
        if (
            parent is None
            or len(parent.get_traits_of_type(trait := InnerRefNamespaceTrait)) != 1
        ):
            raise VerifyException(
                f"Operation {op.name} with trait {type(self).__name__} must have a parent with trait {trait.__name__}"
            )


@dataclass
class InnerSymbolTable:
    """A class for lookups in inner symbol tables. Called InnerSymbolTable in upstream (name clash with trait)."""

    op: InitVar[Operation | None] = None
    symbol_table: dict[StringAttr, InnerSymTarget] = field(default_factory=dict)

    def __post_init__(self, op: Operation | None = None) -> None:
        pass
        # Here will populate self.symbol_table


@dataclass
class InnerSymbolTableCollection:
    """This class represents a collection of InnerSymbolTable."""

    symbol_tables: dict[Operation, InnerSymbolTable] = field(
        default_factory=dict, init=False
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
    """Trait for operations defining a new scope for InnerRef’s. Operations with this trait must be a SymbolTable."""

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
    sym_name: ParameterDef[StringAttr]
    field_id: ParameterDef[IntAttr]
    sym_visibility: ParameterDef[StringAttr]

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
        super().__init__([sym, field_id, sym_visibility])

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
        printer.print_string(str(self.field_id.data))
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

    props: ParameterDef[ArrayAttr[InnerSymPropertiesAttr]]

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
        super().__init__([syms])

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

    INPUT = (0,)
    OUTPUT = (1,)
    INOUT = (2,)

    @staticmethod
    def parse_optional(parser: BaseParser, short: bool = False) -> "Direction | None":
        if parser.parse_optional_keyword("input" if not short else "in"):
            return Direction.INPUT
        elif parser.parse_optional_keyword("output" if not short else "out"):
            return Direction.OUTPUT
        elif parser.parse_optional_keyword("inout"):
            return Direction.INOUT
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
                printer.print("input" if not short else "in")
            case Direction.OUTPUT:
                printer.print("output" if not short else "out")
            case Direction.INOUT:
                printer.print("inout")

    def is_input_like(self) -> bool:
        return self == Direction.INPUT or self == Direction.INOUT


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

    port_name: ParameterDef[StringAttr]
    type: ParameterDef[TypeAttribute]
    dir: ParameterDef[DirectionAttr]


@irdl_attr_definition
class ModuleType(ParametrizedAttribute, TypeAttribute):
    name = "hw.modty"

    ports: ParameterDef[ArrayAttr[ModulePort]]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        def parse_port() -> ModulePort:
            direction = Direction.parse(parser)
            name = parser.parse_identifier("port name")
            parser.parse_punctuation(":")
            typ = parser.parse_type()

            return ModulePort([StringAttr(name), typ, DirectionAttr(direction)])

        return parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parse_port)

    def print_parameters(self, printer: Printer):
        def print_port(port: ModulePort):
            port.dir.data.print(printer)
            printer.print(f" {port.port_name.data} : ")
            printer.print_attribute(port.type)

        with printer.in_angle_brackets():
            printer.print_list(self.ports.data, print_port)


@irdl_attr_definition
class ParamDeclAttr(ParametrizedAttribute):
    name = "hw.param.decl"

    port_name: ParameterDef[StringAttr]
    type: ParameterDef[TypeAttribute]
    value: ParameterDef[Attribute | NoneAttr]  # optional

    @classmethod
    def parse_free_standing_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """
        Parses the parameter declaration without the encompassing angle brackets.
        """

        name = StringAttr(parser.parse_identifier("parameter name"))
        parser.parse_punctuation(":")
        typ = parser.parse_attribute()
        if not isinstance(typ, TypeAttribute):
            parser.raise_error("expected type attribute for parameter")

        value = NoneAttr()
        if parser.parse_optional_punctuation("=") is not None:
            value = parser.parse_attribute()

        return [name, typ, value]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            return cls.parse_free_standing_parameters(parser)

    def print_free_standing_parameters(self, printer: Printer):
        """
        Prints the parameter declaration without the encompassing angle brackets.
        """

        printer.print(f"{self.port_name.data} : ")
        printer.print_attribute(self.type)
        if not isinstance(self.value, NoneAttr):
            printer.print(" = ")
            printer.print_attribute(self.value)

    def print_parameters(self, printer: Printer):
        with printer.in_angle_brackets():
            self.print_free_standing_parameters(printer)


_MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT: list[str] = [
    "sym_name",
    "module_type",
    "parameters",
    "per_port_attrs",
    "comment",
    "result_locs",
]


@irdl_op_definition
class ModuleOp(IRDLOperation):
    """
    Represents a Verilog module, including a given name, a list of ports,
    a list of parameters, and a body that represents the connections within
    the module.
    """

    name = "hw.module"

    sym_name: SymbolNameAttr = attr_def(SymbolNameAttr)
    module_type: ModuleType = attr_def(ModuleType)
    parameters: ArrayAttr[ParamDeclAttr] | None = opt_attr_def(ArrayAttr[ParamDeclAttr])
    per_port_attrs: ArrayAttr[DictionaryAttr] | None = opt_attr_def(
        ArrayAttr[DictionaryAttr]
    )
    comment: StringAttr | None = opt_attr_def(StringAttr)
    result_locs: ArrayAttr[LocationAttr] | None = opt_attr_def(
        ArrayAttr[LocationAttr], attr_name="resultLocs"
    )

    body: SingleBlockRegion = region_def()

    def __init__(
        self,
        sym_name: SymbolNameAttr,
        module_type: ModuleType,
        body: Region | None = None,
        parameters: ArrayAttr[ParamDeclAttr] = ArrayAttr([]),
        per_port_attrs: ArrayAttr[DictionaryAttr] | None = None,
        comment: StringAttr | None = None,
        result_locs: ArrayAttr[LocationAttr] | None = None,
    ):
        attributes: dict[str, Attribute] = {
            "sym_name": sym_name,
            "module_type": module_type,
            "parameters": parameters,
        }

        if per_port_attrs is not None:
            attributes["per_port_attrs"] = per_port_attrs

        if comment is not None:
            attributes["comment"] = comment

        if result_locs is not None:
            attributes["result_locs"] = result_locs

        if body is None:
            body = Region(
                [
                    Block(
                        arg_types=(
                            port.type
                            for port in module_type.ports.data
                            if port.dir.data.is_input_like()
                        )
                    )
                ]
            )

        return super().__init__(
            attributes=attributes,
            regions=[body],
        )

    @classmethod
    def parse(cls, parser: Parser) -> "ModuleOp":
        @dataclass
        class ModuleArg:
            port_dir: Direction
            port_name: str
            port_ssa: Parser.Argument | None
            port_type: TypeAttribute
            port_attrs: dict[str, Attribute]
            port_location: LocationAttr | None

        def parse_optional_port_name() -> str | None:
            name = parser.parse_optional_identifier()
            if name is not None:
                return name
            return parser.parse_optional_str_literal()

        def parse_module_arg() -> ModuleArg:
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

            # TODO BEFORE PR: handle inout type wrapping

            port_attrs = parser.parse_optional_attr_dict()
            port_location = parser.parse_optional_location()

            # Resolve the argument
            if port_ssa is not None:
                port_ssa = Parser.Argument(port_ssa.name, port_type)

            return ModuleArg(
                port_dir, port_name, port_ssa, port_type, port_attrs, port_location
            )

        name = SymbolNameAttr(parser.parse_symbol_name())
        parameters = parser.parse_optional_comma_separated_list(
            parser.Delimiter.ANGLE,
            lambda: ParamDeclAttr(ParamDeclAttr.parse_free_standing_parameters(parser)),
        )
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parse_module_arg
        )
        attrs = parser.parse_optional_attr_dict_with_keyword(
            _MODULE_OP_ATTRS_HANDLED_BY_CUSTOM_FORMAT
        )

        # Extract the ModuleOp attributes and arguments from the parsed arg port list
        region_args: list[Parser.Argument] = []
        module_ports: list[ModulePort] = []
        per_port_attrs: list[DictionaryAttr] = []
        has_non_empty_per_port_attr = False
        result_locs: list[LocationAttr] = []
        has_known_location = False
        for arg in args:
            if arg.port_ssa:
                region_args.append(arg.port_ssa)
            module_ports.append(
                ModulePort(
                    [
                        StringAttr(arg.port_name),
                        arg.port_type,
                        DirectionAttr(arg.port_dir),
                    ]
                )
            )
            per_port_attrs.append(DictionaryAttr(arg.port_attrs))
            has_non_empty_per_port_attr |= len(arg.port_attrs) != 0
            result_locs.append(
                arg.port_location if arg.port_location is not None else LocationAttr()
            )
            has_known_location |= arg.port_location is not None

        body = parser.parse_region(region_args)
        parameters = ArrayAttr(parameters if parameters is not None else [])

        module_op = cls(
            name,
            ModuleType([ArrayAttr(module_ports)]),
            body,
            parameters,
            ArrayAttr(per_port_attrs) if has_non_empty_per_port_attr else None,
            None,
            ArrayAttr(result_locs) if has_known_location else None,
        )

        if attrs is not None:
            for k, v in attrs.data.items():
                module_op.attributes[k] = v

        return module_op

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_attribute(self.sym_name)

        if self.parameters is not None and len(self.parameters.data) != 0:
            with printer.in_angle_brackets():
                printer.print_list(
                    self.parameters.data,
                    lambda x: x.print_free_standing_parameters(printer),
                )

        printer.print("(")
        arg_iter = iter(self.body.block.args)
        result_loc_iter = (
            iter(self.result_locs.data) if self.result_locs is not None else None
        )
        per_port_attrs = (
            iter(self.per_port_attrs) if self.per_port_attrs is not None else None
        )

        def print_port(port: ModulePort):
            ssa_arg = next(arg_iter) if port.dir.data.is_input_like() else None
            port.dir.data.print(printer, short=True)
            printer.print(" ")

            # Print argument
            if ssa_arg is not None:
                used_name = printer.print_ssa_value(ssa_arg)
                if port.port_name.data != used_name:
                    printer.print(" ")
                    printer.print_identifier_or_string_literal(port.port_name.data)
            else:
                printer.print_identifier_or_string_literal(port.port_name.data)
            printer.print(": ")
            printer.print_attribute(port.type)

            # Print port attributes
            if per_port_attrs is not None:
                printer.print_attr_dict(next(per_port_attrs).data)
                printer.print(" ")

            # Print argument location
            if printer.print_debuginfo:
                if ssa_arg is not None:
                    # FIXME: when location is supported in xDSL, fetch location from ssa_arg instead
                    location = LocationAttr()
                elif result_loc_iter is not None:
                    location = next(result_loc_iter)
                else:
                    location = LocationAttr()
                printer.print_attribute(location)

        printer.print_list(self.module_type.ports.data, print_port)

        printer.print(") ")
        printer.print_region(self.body, print_entry_block_args=False)


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "hw.output"

    inputs: VarOperand = var_operand_def(AnyAttr())

    traits = frozenset([IsTerminator(), HasParent(ModuleOp)])

    def __init__(self, ops: Sequence[SSAValue | Operation]):
        super().__init__(operands=ops)

    def verify_(self) -> None:
        parent = self.parent_op()
        assert isinstance(parent, ModuleOp)

        expected_results = [
            port.type
            for port in parent.module_type.ports.data
            if port.dir.data == Direction.OUTPUT
        ]

        if len(expected_results) != len(self.inputs):
            raise VerifyException(
                f"module expected {len(expected_results)} outputs, got {len(self.inputs)}"
            )

        for i, (got, expected) in enumerate(zip(self.inputs, expected_results)):
            if got.type != expected:
                raise VerifyException(
                    f"output {i} is of unexpected type: expected {expected}, got {got.type}"
                )

    @classmethod
    def parse(cls, parser: Parser) -> "OutputOp":
        operands = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_attribute
        )
        operands = parser.resolve_operands(operands, types, parser.pos)
        return cls(operands)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_list(self.inputs, printer.print_operand)
        printer.print(" : ")
        printer.print_list((x.type for x in self.inputs), printer.print_attribute)


HW = Dialect(
    "hw",
    [ModuleOp, OutputOp],
    [
        DirectionAttr,
        InnerRefAttr,
        InnerSymPropertiesAttr,
        InnerSymAttr,
        ModulePort,
        ParamDeclAttr,
    ],
)
