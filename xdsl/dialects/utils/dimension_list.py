from xdsl.dialects.builtin import DenseArrayBase, IntegerType, I64, i64
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


def parse_empty_dimension_list(parser: Parser) -> bool:
    if parser.parse_optional_characters("["):
        if not parser.parse_characters("]"):
            parser.raise_error(
                "Failed parsing dimension list, expected '[]' for an empty list"
            )

        return True

    return False


def print_dimension_list(printer: Printer, dims: DenseArrayBase[IntegerType]) -> None:
    values = dims.get_values()

    if len(values) == 0:
        printer.print_string("[]")
        return

    printer.print_list(
        values,
        lambda x: printer.print_int(x) if x != -1 else printer.print_string("?"),
        "x",
    )


@irdl_custom_directive
class DimensionList(CustomDirective):
    """
    Custom directive for parsing/printing dimension list.

    These look like (e.g.) 3x?x5x2, or [] to denote empty
    """

    dimensions: AttributeVariable

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        # empty list cases is represented with "[]"
        if parse_empty_dimension_list(parser):
            self.dimensions.set(state, DenseArrayBase[I64].from_list(i64, []))

            return True

        # non-empty list case
        dims = parser.parse_dimension_list()

        self.dimensions.set(state, DenseArrayBase[I64].from_list(i64, dims))

        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)

        dims = self.dimensions.get(op)
        assert isa(dims, DenseArrayBase[IntegerType])

        print_dimension_list(printer, dims)
