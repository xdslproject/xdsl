from collections.abc import Sequence
from typing import Generic

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FunctionType,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    AttributeInvT,
    BlockArgument,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import IRDLOperation, var_operand_def
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer


class AbstractYieldOperation(Generic[AttributeInvT], IRDLOperation):
    """
    A base class for yielding operations to inherit, provides the standard custom syntax
    and a definition of the `arguments` variadic operand.
    """

    arguments = var_operand_def(AttributeInvT)

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(operands=[operands])


def print_for_op_like(
    printer: Printer,
    lower_bound: SSAValue,
    upper_bound: SSAValue,
    step: SSAValue,
    iter_args: Sequence[SSAValue],
    body: Region,
    default_indvar_type: type[TypeAttribute] | None = None,
    bound_words: Sequence[str] = ["to"],
):
    """
    Prints the loop bounds, step, iteration arguments, and body.

    Users can provide a default induction variable type and specific human-readable
    words for bounds (default: "to").

    Note that providing a default induction variable type is required to suggest that
    all loop control variable types (induction, bounds and step) have the same type,
    hence moving the induction variable type printing to the end of the for expression.
    The induction variable type printing is ommited when it matches the expected default
    type (`default_indvar_type`).
    """

    block = body.block
    indvar, *block_iter_args = block.args

    printer.print_string(" ")

    def print_indvar_type():
        printer.print_string(" : ")
        printer.print_attribute(indvar.type)
        printer.print_string(" ")

    printer.print_ssa_value(indvar)

    if default_indvar_type is None:
        print_indvar_type()

    printer.print_string(" = ")
    printer.print_ssa_value(lower_bound)

    for word in bound_words:
        printer.print_string(f" {word} ")

    printer.print_ssa_value(upper_bound)
    printer.print_string(" step ")
    printer.print_ssa_value(step)
    printer.print_string(" ")
    if block_iter_args:
        printer.print_string("iter_args(")
        printer.print_list(
            zip(block_iter_args, iter_args),
            lambda pair: print_assignment(printer, *pair),
        )
        printer.print_string(") -> (")
        printer.print_list((a.type for a in block_iter_args), printer.print_attribute)
        printer.print_string(") ")

    if default_indvar_type is not None and not isinstance(
        indvar.type, default_indvar_type
    ):
        print_indvar_type()

    printer.print_region(
        body,
        print_entry_block_args=False,
        print_empty_block=False,
        print_block_terminators=bool(iter_args),
    )


def parse_for_op_like(
    parser: Parser,
    default_indvar_type: TypeAttribute | None = None,
    bound_words: Sequence[str] = ["to"],
) -> tuple[SSAValue, SSAValue, SSAValue, Sequence[SSAValue], Region]:
    """
    Returns the loop bounds, step, iteration arguments, and body.

    Users can provide a default induction variable type and specific human-readable
    words for bounds (default: "to").
    Note that providing a default induction variable type is required to suggest that
    all loop control variable types (induction, bounds and step) have the same type,
    hence the induction variable type is potentially expected at the end of the for
    expression.
    """

    unresolved_indvar = parser.parse_argument(expect_type=False)

    indvar_type = None

    if default_indvar_type is None:
        parser.parse_characters(":")
        indvar_type = parser.parse_type()

    parser.parse_characters("=")
    lower_bound = parser.parse_operand()

    for word in bound_words:
        parser.parse_characters(word)

    upper_bound = parser.parse_operand()
    parser.parse_characters("step")
    step = parser.parse_operand()

    # parse iteration arguments
    pos = parser.pos
    unresolved_iter_args: list[Parser.UnresolvedArgument] = []
    iter_arg_unresolved_operands: list[UnresolvedOperand] = []
    iter_arg_types: list[Attribute] = []
    if parser.parse_optional_characters("iter_args"):
        for iter_arg, iter_arg_operand in parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, lambda: parse_assignment(parser)
        ):
            unresolved_iter_args.append(iter_arg)
            iter_arg_unresolved_operands.append(iter_arg_operand)
        parser.parse_characters("->")
        iter_arg_types = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parser.parse_attribute
        )

    iter_arg_operands = parser.resolve_operands(
        iter_arg_unresolved_operands, iter_arg_types, pos
    )

    # set block argument types
    iter_args = [
        u_arg.resolve(t) for u_arg, t in zip(unresolved_iter_args, iter_arg_types)
    ]

    if default_indvar_type is not None:
        indvar_type = (
            parser.parse_type()
            if parser.parse_optional_characters(":")
            else default_indvar_type
        )
    assert indvar_type is not None

    # set induction variable type
    indvar = unresolved_indvar.resolve(indvar_type)

    body = parser.parse_region((indvar, *iter_args))

    return lower_bound, upper_bound, step, iter_arg_operands, body


def print_func_op_like(
    printer: Printer,
    sym_name: StringAttr,
    function_type: FunctionType,
    body: Region,
    attributes: dict[str, Attribute],
    *,
    arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
    res_attrs: ArrayAttr[DictionaryAttr] | None = None,
    reserved_attr_names: Sequence[str],
):
    printer.print(f" @{sym_name.data}")
    if body.blocks:
        printer.print("(")
        if arg_attrs is not None:
            printer.print_list(
                zip(body.blocks[0].args, arg_attrs),
                lambda arg_with_attrs: print_func_argument(
                    printer, arg_with_attrs[0], arg_with_attrs[1]
                ),
            )
        else:
            printer.print_list(body.blocks[0].args, printer.print_block_argument)
        printer.print(")")
        if function_type.outputs:
            printer.print(" -> ")
            if len(function_type.outputs) > 1 or res_attrs is not None:
                printer.print("(")
            if res_attrs is not None:
                printer.print_list(
                    zip(function_type.outputs, res_attrs),
                    lambda arg_with_attrs: print_func_output(
                        printer, arg_with_attrs[0], arg_with_attrs[1]
                    ),
                )
            else:
                printer.print_list(function_type.outputs, printer.print_attribute)
            if len(function_type.outputs) > 1 or res_attrs is not None:
                printer.print(")")
    else:
        printer.print_attribute(function_type)
    printer.print_op_attributes(
        attributes, reserved_attr_names=reserved_attr_names, print_keyword=True
    )
    printer.print_string(" ", indent=0)

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
    ArrayAttr[DictionaryAttr] | None,
    ArrayAttr[DictionaryAttr] | None,
]:
    """
    Returns the function name, argument types, return types, body, extra args, arg_attrs and res_attrs.
    """
    # Parse function name
    name = parser.parse_symbol_name().data

    def parse_fun_input() -> Attribute | tuple[Parser.Argument, dict[str, Attribute]]:
        arg = parser.parse_optional_argument()
        if arg is None:
            ret = parser.parse_optional_type()
            if ret is None:
                parser.raise_error("Expected argument or type")
        else:
            arg_attr_dict = parser.parse_optional_dictionary_attr_dict()
            ret = (arg, arg_attr_dict)
        return ret

    def parse_fun_output() -> tuple[Attribute, dict[str, Attribute]]:
        arg_type = parser.parse_optional_type()
        if arg_type is None:
            parser.raise_error("Return type should be specified")
        arg_attr_dict = parser.parse_optional_dictionary_attr_dict()
        return (arg_type, arg_attr_dict)

    # Parse function arguments
    args = parser.parse_comma_separated_list(
        parser.Delimiter.PAREN,
        parse_fun_input,
    )

    entry_arg_tuples: list[tuple[Parser.Argument, dict[str, Attribute]]] = []
    input_types: list[Attribute] = []
    for arg in args:
        if isinstance(arg, Attribute):
            input_types.append(arg)
        else:
            entry_arg_tuples.append(arg)

    if entry_arg_tuples:
        # Check consistency (They should be either all named or none)
        if input_types:
            parser.raise_error(
                "Expected all arguments to be named or all arguments to be unnamed."
            )

        entry_args = [arg for arg, _ in entry_arg_tuples]
        input_types = [arg.type for arg in entry_args]
    else:
        entry_args = None

    if any(attrs for _, attrs in entry_arg_tuples):
        arg_attrs = ArrayAttr(DictionaryAttr(attrs) for _, attrs in entry_arg_tuples)
    else:
        arg_attrs = None

    # Parse return type
    return_types: list[Attribute] = []
    res_attrs_raw: list[dict[str, Attribute]] | None = []
    if parser.parse_optional_punctuation("->"):
        return_attributes = parser.parse_optional_comma_separated_list(
            parser.Delimiter.PAREN, parse_fun_output
        )
        if return_attributes is None:
            # output attributes are supported only if return results are enclosed in brackets (...)
            return_types, res_attrs_raw = [parser.parse_type()], None
        else:
            return_types, res_attrs_raw = (
                [el[0] for el in return_attributes],
                [el[1] for el in return_attributes],
            )

    if res_attrs_raw is not None and any(res_attrs_raw):
        res_attrs = ArrayAttr(DictionaryAttr(attrs) for attrs in res_attrs_raw)
    else:
        res_attrs = None

    extra_attributes = parser.parse_optional_attr_dict_with_keyword(reserved_attr_names)

    # Parse body
    region = parser.parse_optional_region(entry_args)
    if region is None:
        region = Region()

    return (
        name,
        input_types,
        return_types,
        region,
        extra_attributes,
        arg_attrs,
        res_attrs,
    )


def print_func_argument(
    printer: Printer, arg: BlockArgument, attrs: DictionaryAttr | None
):
    printer.print_block_argument(arg)
    if attrs is not None and attrs.data:
        printer.print_op_attributes(attrs.data)


def print_func_output(
    printer: Printer, out_type: Attribute, attrs: DictionaryAttr | None
):
    printer.print_attribute(out_type)
    if attrs is not None and attrs.data:
        printer.print_op_attributes(attrs.data)


def print_assignment(printer: Printer, arg: BlockArgument, val: SSAValue):
    printer.print_block_argument(arg, print_type=False)
    printer.print_string(" = ")
    printer.print_ssa_value(val)


def parse_assignment(
    parser: Parser,
) -> tuple[Parser.UnresolvedArgument, UnresolvedOperand]:
    arg = parser.parse_argument(expect_type=False)
    parser.parse_characters("=")
    val = parser.parse_unresolved_operand()
    return arg, val
