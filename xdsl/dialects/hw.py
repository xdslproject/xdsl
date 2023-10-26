"""
This is a stub of CIRCT’s hw dialect.
It currently implements minimal types and operations used by other dialects.

[1] https://circt.llvm.org/docs/Dialects/HW/
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import cast, overload

from xdsl.dialects.builtin import (
    ArrayAttr,
    FlatSymbolRefAttr,
    IntAttr,
    ParameterDef,
    StringAttr,
)
from xdsl.ir import (
    Dialect,
    Operation,
    OpResult,
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


class FieldIDTypeInterface:
    """Common methods for types which can be indexed by a field_id.

    field_id is a depth-first numbering of the elements of a type. For example:
    ```
    struct a  /* 0 */ {
      int b; /* 1 */
      struct c /* 2 */ {
        int d; /* 3 */
      }
    }

    int e; /* 0 */
    ```
    """

    def get_max_field_id(self) -> int:
        """Get the maximum field ID for this type"""
        return 0

    def get_sub_type_by_field_id(self, field_id: int) -> tuple[type, int]:
        """Get the sub-type of a type for a field ID, and the subfield's ID. Strip
        off a single layer of this type and return the sub-type and a field ID
        targeting the same field, but rebased on the sub-type.

        The resultant type *may* not be a FieldIDTypeInterface if the resulting
        field_id is zero. This means that leaf types may be ground without
        implementing an interface. An empty aggregate will also appear as a zero."""
        if field_id == 0:
            return (type(self), 0)
        raise NotImplementedError()

    def project_to_child_field_id(self, field_id: int, index: int) -> tuple[int, bool]:
        """Returns the effective field id when treating the index field as the
        root of the type. Essentially maps a field_id to a field_id after a
        subfield op. Returns the new id and whether the id is in the given
        child."""
        ...

    def get_index_for_field_id(self, field_id: int) -> int:
        """Returns the index (e.g. struct or vector element) for a given
        field_id. This returns the containing index in the case that the
        field_id points to a child field of a field."""
        ...

    def get_field_id(self, index: int) -> int:
        """Return the field_id of a given index (e.g. struct or vector element).
        field ids start at 1, and are assigned to each field in a recursive
        depth-first walk of all elements. A field ID of 0 is used to reference
        the type itself."""
        ...

    def get_index_and_subfield_id(self, field_id: int) -> tuple[int, int]:
        """Find the index of the element that contains the given field_id. As well, rebase the field_id to the element."""
        ...


class InnerSymTarget:
    """The target of an inner symbol, the entity the symbol is a handle for."""

    def __init__(
        self,
        op: Operation | None = None,
        field_id: int = 0,
        port_idx: int | None = None,
    ) -> None:
        self.op = op
        self.field_id = field_id
        self.port_idx = port_idx

    def __bool__(self):
        # None-valued op defines an invalid target
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
        return cls(base.op, base.field_id + field_id, base.port_idx)


@irdl_attr_definition
class InnerRefAttr(ParametrizedAttribute):
    """This works like a symbol reference, but to a name inside a module.

    NB: the parse and print for AsmPrinter are not copied from CIRCT."""

    name = "hw.inner_name_ref"
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


class InnerSymbolTable(OpTrait):
    """A trait for inner symbol table functionality on an operation.
    Merges the upstream table of inner symbols and their resolutions and the op trait.
    """

    def __init__(
        self, table: Mapping[StringAttr, InnerSymTarget] | None = None
    ) -> None:
        if table is None:
            table = dict()
        self.symbol_table = table
        super().__init__()

    @classmethod
    def verify_region_trait(cls, op: Operation):
        # Insist that ops with InnerSymbolTable's provide a Symbol, this is
        # essential to how InnerRef's work.
        if not op.has_trait(SymbolOpInterface):
            raise VerifyException("expected operation to define a Symbol")

        # InnerSymbolTable's must be directly nested within an InnerRefNamespace.
        # NB: upstream uses InnerRefNamespaceLike
        parent = op.parent_op()
        if parent is None or not parent.has_trait(InnerRefNamespace):
            raise VerifyException("InnerSymbolTable must have InnerRefNamespace parent")

    def lookup(
        self, inner_symbol_table_op: Operation, name: StringAttr
    ) -> InnerSymTarget:
        """Look up a symbol with the specified name, returning empty InnerSymTarget if
        no such name exists. Names never include the @ on them."""
        return self.symbol_table.get(name, InnerSymTarget())

    def lookup_op(
        self, inner_symbol_table_op: Operation, name: StringAttr
    ) -> Operation | None:
        """Look up a symbol with the specified name, returning null if no such name exists or doesn't target just an operation."""
        result = self.lookup(inner_symbol_table_op, name)
        if result.is_op_only():
            result.op


class InnerSymbolTableCollection:
    """This class represents an InnerSymbolTable collection. Simplified from upstream
    to remove mapping from operations to their traits."""

    def __init__(self, inner_ref_ns_op: Operation | None = None) -> None:
        self.symbol_tables: set[Operation] = set()
        if inner_ref_ns_op is not None:
            self.populate_and_verify_tables(inner_ref_ns_op)

    def get_inner_symbol_table(self, op: Operation) -> InnerSymbolTable:
        """Returns the InnerSymolTable trait, ensuring it is in the collection"""
        table = op.get_trait(InnerSymbolTable)
        if table is None:
            raise VerifyException(f"Operation {op} should have InenrSymbolTable trait")
        if op not in self.symbol_tables:
            self.symbol_tables.add(op)
        return table

    def populate_and_verify_tables(self, inner_ref_ns_op: Operation):
        """Populate tables for all InnerSymbolTable operations in the given InnerRefNamespace operation, verifying each."""
        # Gather top-level operations that have the InnerSymbolTable trait.
        inner_sym_table_ops = (
            op for op in inner_ref_ns_op.walk() if op.has_trait(InnerSymbolTable)
        )

        # Construct the tables
        for op in inner_sym_table_ops:
            if op in self.symbol_tables:
                raise VerifyException(
                    f"Trying to insert the same op {op} twice in symbol tables"
                )
            self.symbol_tables.add(op)


class InnerRefUserOpInterface(OpTrait):
    """This interface describes an operation that may use a `InnerRef`. This
    interface allows for users of inner symbols to hook into verification and
    other inner symbol related utilities that are either costly or otherwise
    disallowed within a traditional operation."""

    def verify_inner_refs(self, op: Operation, namespace: "InnerRefNamespace"):
        """Verify the inner ref uses held by this operation."""
        ...


class InnerRefNamespace(OpTrait):
    parameters: tuple[SymbolTable, InnerSymbolTableCollection]

    def __init__(
        self, symbol_table: SymbolTable, inner_sym_tables: InnerSymbolTableCollection
    ):
        super().__init__([symbol_table, inner_sym_tables])

    @property
    def symbol_table(self) -> SymbolTable:
        return self.parameters[0]

    @property
    def inner_sym_tables(self) -> InnerSymbolTableCollection:
        return self.parameters[1]

    @classmethod
    def verify_region_trait(cls, op: Operation):
        if not op.has_trait(SymbolTable):
            raise VerifyException("expected operation to be a SymbolTable")

        if len(op.regions) != 1:
            raise VerifyException("expected operation to have a single region")
        if len(op.regions[0].blocks) != 1:
            raise VerifyException("expected operation to have a single block")

        inner_sym_tables = InnerSymbolTableCollection()
        inner_sym_tables.populate_and_verify_tables(op)
        symbol_table = SymbolTable(op)
        namespace = InnerRefNamespace(symbol_table, inner_sym_tables)

        for inner_op in op.walk():
            inner_ref_user_op_trait = inner_op.get_trait(InnerRefUserOpInterface)
            if inner_ref_user_op_trait is not None:
                inner_ref_user_op_trait.verify_inner_refs(inner_op, namespace)

    def lookup(self, op: Operation, inner: InnerRefAttr) -> InnerSymTarget | None:
        module = self.symbol_table.lookup_symbol(op, inner.get_module())
        if module is None:
            return None
        if not module.has_trait(InnerSymbolTable):
            raise VerifyException("module should implement inner symbol table")
        table = self.inner_sym_tables.get_inner_symbol_table(module)
        return table.lookup(module, inner.sym_name)

    def lookup_op(self, op: Operation, inner: InnerRefAttr) -> Operation | None:
        module = self.symbol_table.lookup_symbol(op, inner.get_module())
        if module is None:
            return None
        if not module.has_trait(InnerSymbolTable):
            raise VerifyException("module should implement inner symbol table")
        table = self.inner_sym_tables.get_inner_symbol_table(module)
        return table.lookup_op(module, inner.sym_name)


@irdl_attr_definition
class InnerSymPropertiesAttr(ParametrizedAttribute):
    name = "hw.inner_sym_props"

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
    def parse_parameter(cls, parser: AttrParser) -> tuple[str, int, str]:
        parser.parse_punctuation("<")
        sym_name = parser.parse_identifier()
        parser.parse_punctuation(",")
        field_id = parser.parse_integer(allow_negative=False, allow_boolean=False)
        parser.parse_punctuation(",")
        sym_visibility = parser.parse_str_literal()
        if sym_visibility not in {"public", "private", "nested"}:
            parser.raise_error('expected "public", "private", or "nested"')
        parser.parse_punctuation(">")
        return (sym_name, field_id, sym_visibility)

    def print_parameter(self, printer: Printer) -> None:
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
class InnerSymAttr(ParametrizedAttribute, Iterable[InnerSymPropertiesAttr]):
    """Inner symbol definition

    Defines the properties of an inner_sym attribute. It specifies the symbol
    name and symbol visibility for each field ID. For any ground types,
    there are no subfields and the field ID is 0. For aggregate types, a
    unique field ID is assigned to each field by visiting them in a
    depth-first pre-order. The custom assembly format ensures that for ground
    types, only `@<sym_name>` is printed.
    """

    props: ParameterDef[ArrayAttr[InnerSymPropertiesAttr]]

    @overload
    def __init__(self) -> None:
        # Create an empty array, represents an invalid InnerSym.
        ...

    @overload
    def __init__(self, syms: str | StringAttr) -> None:
        ...

    @overload
    def __init__(
        self, syms: Sequence[InnerSymPropertiesAttr] | ArrayAttr[InnerSymPropertiesAttr]
    ) -> None:
        ...

    def __init__(
        self,
        syms: str
        | StringAttr
        | Sequence[InnerSymPropertiesAttr]
        | ArrayAttr[InnerSymPropertiesAttr] = [],
    ) -> None:
        if isinstance(syms, str | StringAttr):
            syms = [InnerSymPropertiesAttr(syms)]
        if not isinstance(syms, ArrayAttr):
            syms = ArrayAttr(syms)
        super().__init__([syms])

    def get_sym_if_exists(self, field_id: int) -> StringAttr | None:
        """Get the inner sym name for field_id, if it exists."""
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

    def erase(self, field_id: int) -> "InnerSymAttr":
        """Return an InnerSymAttr with the inner symbol for the specified field_id removed."""
        return InnerSymAttr([prop for prop in self.props if prop.field_id != field_id])

    def verify(self):
        pass

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[InnerSymPropertiesAttr, ...]:
        data = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_attribute
        )
        # the type system can't ensure that the elements are of type InnerSymPropertiesAttr
        return cast(tuple[InnerSymPropertiesAttr, ...], tuple(data))

    def print_parameter(self, printer: Printer):
        if (
            len(self) == 1
            and (sym_name := self.get_sym_name()) is not None
            and self.props.data[0].sym_visibility.data == "public"
        ):
            printer.print_string("@")
            printer.print_string(sym_name.data)
        else:
            printer.print_string("[")
            for n, prop in enumerate(
                sorted(self.props, key=lambda prop: prop.field_id.data)
            ):
                if n:
                    printer.print_string(",")
                printer.print_attribute(prop.sym_name)
            printer.print_string("]")


class InnerSymbolOpInterface(OpTrait):
    """
    This interface describes an operation that may define an `inner_sym`. An
    `inner_sym` operation resides in arbitrarily-nested regions of a region that
    defines a `InnerSymbolTable`. Inner Symbols are different from normal
    symbols due to MLIR symbol table resolution rules.

    https://circt.llvm.org/docs/Dialects/HW/#innersymbolopinterface-innersymbol
    """

    def get_inner_name_attr(self, op: Operation) -> StringAttr | None:
        """Returns the name of the top-level inner symbol defined by this operation, if present."""
        inner_sym = self.get_inner_sym_attr(op)
        if inner_sym is None:
            return None
        return inner_sym.get_sym_name()

    def get_inner_name(self, op: Operation) -> str | None:
        """Returns the name of the top-level inner symbol defined by this operation, if present."""
        attr = self.get_inner_name_attr(op)
        if attr is not None:
            return attr.name

    def set_inner_symbol(self, op: Operation, name: StringAttr):
        """Sets the name of the top-level inner symbol defined by this operation to the specified string, dropping any symbols on fields."""
        op.attributes["inner_sym"] = InnerSymAttr(name)

    def set_inner_symbol_attr(self, op: Operation, sym: InnerSymAttr | None):
        """Sets the inner symbols defined by this operation."""
        if sym is not None:
            op.attributes["inner_sym"] = sym
        elif "inner_sym" in op.attributes:
            del op.attributes["inner_sym"]

    def get_inner_ref(self, op: Operation) -> InnerRefAttr:
        """Returns an InnerRef to this operation's top-level inner symbol, which must be present."""
        if (parent := op.get_parent_with_trait(InnerSymbolTable)) is None:
            raise VerifyException
        if (module_name := self.get_inner_name_attr(parent)) is None:
            raise VerifyException
        if (sym_name := self.get_inner_name_attr(op)) is None:
            raise VerifyException
        return InnerRefAttr(module_name, sym_name)

    def get_inner_sym_attr(self, op: Operation) -> InnerSymAttr | None:
        """Returns the InnerSymAttr representing all inner symbols defined by this operation."""
        inner_sym = op.attributes.get("inner_sym", None)
        if inner_sym is None:
            return None
        if not isinstance(inner_sym, InnerSymAttr):
            raise VerifyException(
                f'Operation {op.name} must have a "inner_sym" attribute of type '
                f"`InnerSymAttr` to conform to {InnerSymbolOpInterface.__name__}"
            )
        return inner_sym

    @classmethod
    def supports_per_field_symbols(cls, op: Operation) -> bool:
        """Returns whether per-field symbols are supported for this operation type."""
        return cls.get_target_result_index(op) is not None

    @classmethod
    def get_target_result_index(cls, op: Operation) -> int | None:
        """Returns the index of the result the innner symbol targets, if applicable. Per-field symbols are resolved into this."""
        inner_symbol_trait = op.get_trait(InnerSymbolOpInterface)

        if inner_symbol_trait is not None:
            return inner_symbol_trait.get_target_result_index(op)

    def get_target_result(self, op: Operation) -> OpResult | None:
        """Returns the result the innner symbol targets, if applicable. Per-field symbols are resolved into this."""
        index = InnerSymbolOpInterface.get_target_result_index(op)
        if index is not None:
            return op.results[index]

    def verify(self, op: Operation) -> None:
        """Check that the operation satisfies the trait requirements."""
        inner_sym = self.get_inner_sym_attr(op)
        if inner_sym is None:
            return
        if not len(inner_sym):
            raise VerifyException("has empty list of inner symbols")

        if not self.supports_per_field_symbols(op):
            # The inner sym can only be specified on field_id=0.
            if len(inner_sym) > 1 or not inner_sym.get_sym_name():
                raise VerifyException("does not support per-field inner symbols")
            return

        result = self.get_target_result(op)
        # If op supports per-field symbols, but does not have a target result, its up to the operation to verify itself.
        # (there are no uses for this presently, but be open to this anyway.)
        if not result:
            return

        # Ensure field_id and symbol names are unique.
        result_type = result.type
        if not isinstance(result_type, FieldIDTypeInterface):
            raise VerifyException(
                "per-field symbol support requires types implementing FieldIDTypeInterface"
            )

        max_fields = result_type.get_max_field_id()
        indices: set[int] = set()
        names: set[str] = set()
        for prop in inner_sym:
            if prop.field_id.data > max_fields:
                raise VerifyException(
                    f'field id:"{prop.field_id.data}" is greater than the maximum field id:"{max_fields}"'
                )
            if prop.field_id.data in indices:
                raise VerifyException(
                    f'cannot assign multiple symbol names to the field id:"{prop.field_id.data}"'
                )
            if prop.sym_name.data in names:
                raise VerifyException(
                    f'cannot reuse symbol name:"{prop.sym_name.data}"'
                )
            indices.add(prop.field_id.data)
            names.add(prop.sym_name.data)


HW = Dialect(
    [],
    [
        InnerRefAttr,
        InnerSymPropertiesAttr,
        InnerSymAttr,
    ],
)
