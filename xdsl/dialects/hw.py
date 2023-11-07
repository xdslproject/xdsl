"""
This is a stub of CIRCT’s hw dialect.
It currently implements minimal types and operations used by other dialects.

[1] https://circt.llvm.org/docs/Dialects/HW/
"""

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
