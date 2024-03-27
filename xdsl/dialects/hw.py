"""
This is a stub of CIRCT’s hw dialect.
It currently implements minimal types and operations for the symbols and inner symbols used by other dialects.

[1] https://circt.llvm.org/docs/Dialects/HW/
[2] https://circt.llvm.org/docs/RationaleSymbols/
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import InitVar, dataclass, field
from typing import overload

from xdsl.dialects.builtin import (
    ArrayAttr,
    FlatSymbolRefAttr,
    IntAttr,
    ParameterDef,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    OpaqueSyntaxAttribute,
    Operation,
    ParametrizedAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
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


HW = Dialect(
    "hw",
    [],
    [
        InnerRefAttr,
        InnerSymPropertiesAttr,
        InnerSymAttr,
    ],
)
