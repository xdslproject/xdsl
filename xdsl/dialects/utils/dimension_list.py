from xdsl.dialects.builtin import I64, DenseArrayBase, IntegerType, i64
from xdsl.irdl.declarative_assembly_format import (
    AttributeVariable,
    CustomDirective,
    IRDLOperation,
    ParsingState,
    PrintingState,
    irdl_custom_directive,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.hints import isa


def parse_empty_dimension_list_directive(parser: Parser) -> bool:
    if parser.parse_optional_characters("["):
        parser.parse_characters("]")
        return True

    return False


def print_empty_dimension_list_directive(printer: Printer) -> None:
    printer.print_string("[]")


@irdl_custom_directive
class DimensionList(CustomDirective):
    """
    Custom directive for parsing/printing dimension list.

    These look like (e.g.) 3x?x5x2, or [] to denote empty
    """
    dimensions: AttributeVariable

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        dims = []

        if parse_empty_dimension_list_directive(parser):
            dims = parser.parse_dimension_list()

        self.dimensions.set(state, DenseArrayBase[I64].from_list(i64, dims))

        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)

        dims = self.dimensions.get(op)
        assert isa(dims, DenseArrayBase[IntegerType])
        values = dims.get_values()

        if not values:
            print_empty_dimension_list_directive(printer)
            return

        printer.print_dimension_list(values)

