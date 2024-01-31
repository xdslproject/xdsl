"""
This is a stub of CIRCT’s hw dialect.
It currently implements minimal types and operations used by other dialects.

[1] https://circt.llvm.org/docs/Dialects/HW/
"""

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    FlatSymbolRefAttr,
    ParameterDef,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


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


HW = Dialect(
    "hw",
    [],
    [
        InnerRefAttr,
    ],
)
