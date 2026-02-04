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


class AbstractYieldOperation(IRDLOperation, Generic[AttributeInvT]):
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


def _print_func_outputs(
    printer: Printer,
    outputs: Sequence[Attribute],
    res_attrs: ArrayAttr[DictionaryAttr] | None,
):
    """
    Print function output types with optional result attributes.

    Supports the following syntax:
        - `-> type` for a single result without attributes
        - `-> (type1, type2, ...)` for multiple results
        - `-> (type {attr = value}, ...)` for results with attributes
    """
    if not outputs:
        return

    printer.print_string(" -> ")

    # parens for multiple outputs or attrs
    needs_parens = len(outputs) > 1 or res_attrs is not None
    if needs_parens:
        printer.print_string("(")

    # print types, optionally paired with their attributes
    if res_attrs is None:
        printer.print_list(outputs, printer.print_attribute)
    else:
        printer.print_list(
            zip(outputs, res_attrs),
            lambda t: print_func_output(printer, t[0], t[1]),
        )

    if needs_parens:
        printer.print_string(")")


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
    printer.print_string(" ")
    printer.print_symbol_name(sym_name.data)
    if body.blocks:
        with printer.in_parens():
            if arg_attrs is not None:
                printer.print_list(
                    zip(body.blocks[0].args, arg_attrs),
                    lambda arg_with_attrs: print_func_argument(
                        printer, arg_with_attrs[0], arg_with_attrs[1]
                    ),
                )
            else:
                printer.print_list(body.blocks[0].args, printer.print_block_argument)

        _print_func_outputs(printer, function_type.outputs.data, res_attrs)
    else:
        printer.print_attribute(function_type)
    printer.print_op_attributes(
        attributes, reserved_attr_names=reserved_attr_names, print_keyword=True
    )
    printer.print_string(" ", indent=0)

    if body.blocks:
        printer.print_region(body, False, False)


def _parse_func_outputs(
    parser: Parser,
) -> tuple[list[TypeAttribute], ArrayAttr[DictionaryAttr] | None]:
    """
    Inverse of `_print_func_outputs`.

    Returns a tuple of (return_types, res_attrs). If there are no return types,
    returns ([], None).
    """
    # no arrow implies no return types
    if not parser.parse_optional_punctuation("->"):
        return [], None

    # attrs only supported with parens
    results = parser.parse_optional_comma_separated_list(
        parser.Delimiter.PAREN,
        lambda: (parser.parse_type(), parser.parse_optional_dictionary_attr_dict()),
    )

    # no parens implies single type without attrs
    if results is None:
        return [parser.parse_type()], None

    # empty parens
    if not results:
        return [], None

    # unpack types and attrs, wrap attrs in ArrayAttr
    types, attrs_raw = zip(*results)
    has_attrs = any(attrs_raw)
    res_attrs = ArrayAttr(DictionaryAttr(a) for a in attrs_raw) if has_attrs else None
    return list(types), res_attrs


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

    # Parse return types
    return_types, res_attrs = _parse_func_outputs(parser)

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
