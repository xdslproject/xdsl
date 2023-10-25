from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import (
    DictionaryAttr,
    FunctionType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Attribute, BlockArgument, Operation, Region, SSAValue
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.hints import isa


def print_call_op_like(
    printer: Printer,
    op: Operation,
    callee: SymbolRefAttr,
    args: Sequence[SSAValue],
    attributes: dict[str, Attribute],
    *,
    reserved_attr_names: Sequence[str],
):
    printer.print_string(" ")
    printer.print_attribute(callee)
    printer.print_string("(")
    printer.print_list(args, printer.print_ssa_value)
    printer.print_string(")")
    printer.print_op_attributes(attributes, reserved_attr_names=reserved_attr_names)
    printer.print_string(" : ")
    printer.print_operation_type(op)


def parse_call_op_like(
    parser: Parser, *, reserved_attr_names: Sequence[str]
) -> tuple[
    SymbolRefAttr, Sequence[SSAValue], Sequence[Attribute], DictionaryAttr | None
]:
    callee = parser.parse_symbol_name()
    unresolved_arguments = parser.parse_op_args_list()
    extra_attributes = parser.parse_optional_attr_dict_with_reserved_attr_names(
        reserved_attr_names
    )
    parser.parse_characters(":")
    pos = parser.pos
    function_type = parser.parse_function_type()
    arguments = parser.resolve_operands(
        unresolved_arguments, function_type.inputs.data, pos
    )
    return (
        SymbolRefAttr(callee),
        arguments,
        function_type.outputs.data,
        extra_attributes,
    )


def print_return_op_like(
    printer: Printer, attributes: dict[str, Attribute], arguments: Sequence[SSAValue]
) -> None:
    if attributes:
        printer.print(" ")
        printer.print_op_attributes(attributes)

    if arguments:
        printer.print(" ")
        printer.print_list(arguments, printer.print_ssa_value)
        printer.print(" : ")
        printer.print_list((x.type for x in arguments), printer.print_attribute)


def parse_return_op_like(
    parser: Parser,
) -> tuple[dict[str, Attribute], Sequence[SSAValue]]:
    attrs = parser.parse_optional_attr_dict()

    pos = parser.pos
    unresolved_operands = parser.parse_optional_undelimited_comma_separated_list(
        parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
    )

    args: Sequence[SSAValue]
    if unresolved_operands is not None:
        parser.parse_punctuation(":")
        types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_type, "Expected return value type"
        )
        args = parser.resolve_operands(unresolved_operands, types, pos)
    else:
        args = ()

    return attrs, args


def print_func_op_like(
    printer: Printer,
    sym_name: StringAttr,
    function_type: FunctionType,
    body: Region,
    attributes: dict[str, Attribute],
    *,
    reserved_attr_names: Sequence[str],
):
    printer.print(f" @{sym_name.data}")
    if body.blocks:
        printer.print("(")
        printer.print_list(body.blocks[0].args, printer.print_block_argument)
        printer.print(") ")
        if function_type.outputs:
            printer.print("-> ")
            if len(function_type.outputs) > 1:
                printer.print("(")
            printer.print_list(function_type.outputs, printer.print_attribute)
            if len(function_type.outputs) > 1:
                printer.print(")")
            printer.print(" ")
    else:
        printer.print_attribute(function_type)
    printer.print_op_attributes(
        attributes, reserved_attr_names=reserved_attr_names, print_keyword=True
    )

    if body.blocks:
        printer.print_region(body, False, False)


def parse_func_op_like(
    parser: Parser, *, reserved_attr_names: Sequence[str]
) -> tuple[
    str,
    Sequence[Attribute],
    Sequence[Attribute],
    Region,
    DictionaryAttr | None,
]:
    # Parse function name
    name = parser.parse_symbol_name().data

    def parse_fun_input():
        ret = parser.parse_optional_argument()
        if ret is None:
            ret = parser.parse_optional_type()
        if ret is None:
            parser.raise_error("Expected argument or type")
        return ret

    # Parse function arguments
    args = parser.parse_comma_separated_list(
        parser.Delimiter.PAREN,
        parse_fun_input,
    )

    # Check consistency (They should be either all named or none)
    if isa(args, list[parser.Argument]):
        entry_args = args
        input_types = cast(list[Attribute], [a.type for a in args])
    elif isa(args, list[Attribute]):
        entry_args = None
        input_types = args
    else:
        parser.raise_error(
            "Expected all arguments to be named or all arguments to be unnamed."
        )

    # Parse return type
    if parser.parse_optional_punctuation("->"):
        return_types = parser.parse_optional_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_type
        )
        if return_types is None:
            return_types = [parser.parse_type()]
    else:
        return_types = []

    extra_attributes = parser.parse_optional_attr_dict_with_keyword(reserved_attr_names)

    # Parse body
    region = parser.parse_optional_region(entry_args)
    if region is None:
        region = Region()

    return name, input_types, return_types, region, extra_attributes


def print_assignment(printer: Printer, arg: BlockArgument, val: SSAValue):
    printer.print_block_argument(arg, print_type=False)
    printer.print_string(" = ")
    printer.print_ssa_value(val)


def parse_assignment(parser: Parser) -> tuple[Parser.Argument, UnresolvedOperand]:
    arg = parser.parse_argument(expect_type=False)
    parser.parse_characters("=")
    val = parser.parse_unresolved_operand()
    return arg, val
