"""
The Data Layout and Target Information (DLTI) dialect is intended to
hold attributes and other components pertaining to descriptions of
in-memory data layout and compilation targets.

https://mlir.llvm.org/docs/Dialects/DLTIDialect/
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class DataLayoutEntryAttr(ParametrizedAttribute):
    """
    An attribute to represent an entry of a data layout specification.
    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#datalayoutentryattr
    """

    name = "dlti.dl_entry"

    key: Attribute
    value: Attribute

    def verify(self) -> None:
        if not isinstance(self.key, StringAttr | TypeAttribute):
            raise VerifyException("key must be a string or a type attribute")


@irdl_attr_definition
class DataLayoutSpecAttr(ParametrizedAttribute):
    """
    An attribute to represent a data layout specification.

    A data layout specification is a list of entries that specify (partial) data
    layout information. It is expected to be attached to operations that serve as
    scopes for data layout requests.

    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#datalayoutspecattr
    """

    name = "dlti.dl_spec"

    # In MLIR, this is a DataLayoutEntryInterface.
    entries: ArrayAttr[DataLayoutEntryAttr]

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> tuple[ArrayAttr[DataLayoutEntryAttr]]:
        def parse_entry() -> DataLayoutEntryAttr:
            entry = parser.parse_attribute()
            parser.parse_punctuation("=")
            value = parser.parse_attribute()
            return DataLayoutEntryAttr(entry, value)

        entries = parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parse_entry)

        return (ArrayAttr(entries),)

    def print_parameters(self, printer: Printer):
        def print_entry(entry: DataLayoutEntryAttr):
            printer.print_attribute(entry.key)
            printer.print_string(" = ")
            printer.print_attribute(entry.value)

        with printer.in_angle_brackets():
            printer.print_list(self.entries, print_entry)


DLTI = Dialect("dlti", [], [DataLayoutEntryAttr, DataLayoutSpecAttr])
