"""
This file contains utility functions to manipulate lists of integers and SSAValues,
that are encoded as a dense list of integers attributes and a variadic operand.

For instance, the `[1, 2, %3, 4, %5]` list is represented as:
- `static_values = [1, 2, MARKER, 4, MARKER]`
- `dynamic_values = [%3, %5]`
where `MARKER` is a special value that indicates that the value at that position is dynamic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from xdsl.ir import Attribute, SSAValue
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


def verify_dynamic_index_list(
    static_values: Sequence[int],
    dynamic_values: Sequence[SSAValue],
    dynamic_index: int,
    end_message: str = "",
) -> None:
    """
    Verify that the static list contains only integers and that the dynamic list
    contains only SSAValues. The static list should be of the same length as the
    dynamic list, with the dynamic values replaced by `dynamic_index`.
    """
    num_dyn_indices = tuple(static_values).count(dynamic_index)
    if num_dyn_indices != len(dynamic_values):
        raise VerifyException(
            "The number of dynamic positions passed as values "
            f"({len(dynamic_values)}) does not match "
            "the number of dynamic position markers "
            f"({num_dyn_indices}){end_message}."
        )


def get_dynamic_index_list(
    static_values: Sequence[int],
    dynamic_values: Sequence[SSAValue],
    dynamic_index: int,
) -> list[SSAValue | int]:
    """
    Get a mixed list of SSAValues and integers from the static and dynamic
    lists. The static list contains integers, where the ones marked with
    `dynamic_index` are to be replaced with the SSAValues from the dynamic list.

    For instance, given the static list `[1, 2, MARKER, 4, MARKER]` and the
    dynamic list `[%3, %5]`, the result will be `[1, 2, %3, 4, %5]`.
    """
    next_dynamic_index = 0
    result: list[SSAValue | int] = []
    for pos in static_values:
        assert isinstance(pos, int)
        if pos == dynamic_index:
            result.append(dynamic_values[next_dynamic_index])
            next_dynamic_index += 1
            continue
        result.append(pos)
    return result


def split_dynamic_index_list(
    values: Sequence[SSAValue | int],
    dynamic_index: int,
) -> tuple[list[int], list[SSAValue]]:
    """
    Split a mixed list of SSAValues and integers into static and dynamic lists.
    The static list contains integers, where the ones marked with `dynamic_index`
    are to be replaced with the SSAValues from the dynamic list.

    For instance, given the list `[1, 2, %3, 4, %5]`, the result will be:
    - static: `[1, 2, MARKER, 4, MARKER]`
    - dynamic: [%3, %5]
    """
    static_values = [
        value if isinstance(value, int) else dynamic_index for value in values
    ]
    dynamic_values = [value for value in values if isinstance(value, SSAValue)]
    return static_values, dynamic_values


def print_dynamic_index_list(
    printer: Printer,
    dynamic_index: int,
    values: Sequence[SSAValue],
    integers: Iterable[int],
    value_types: Sequence[Attribute] = (),
    *,
    delimiter: Parser.Delimiter = Parser.Delimiter.SQUARE,
):
    """
    Prints a list with either
      1) the static integer value in `integers` is `ShapedType.DYNAMIC` or
      2) the next value otherwise.
    If `value_types` is non-empty, it is expected to contain as many elements as `values`
    indicating their types. This allows idiomatic printing of mixed value and
    integer attributes in a list. E.g.
    `[%arg0 : index, 7, 42, %arg42 : i32]`.
    The `dynamic_index` parameter specifies the sentinel value to use to print an ssa
    value instead of a constant.
    """
    if delimiter.value is not None:
        printer.print_string(delimiter.value[0])
    value_index = 0
    for integer_index, integer in enumerate(integers):
        if integer_index:
            printer.print_string(", ")
        if integer == dynamic_index:
            printer.print_ssa_value(values[value_index])
            if value_types:
                printer.print_string(" : ")
                printer.print_attribute(value_types[value_index])
            value_index += 1
        else:
            printer.print_string(f"{integer}")
    if delimiter.value is not None:
        printer.print_string(delimiter.value[1])


def parse_dynamic_index_with_type(parser: Parser) -> int | SSAValue:
    """
    Parses an element in an index list, either an index or an ssa value with a type:
    e.g. `42` or `%val : i32`.
    """
    value = parser.parse_optional_unresolved_operand()
    if value is None:
        return parser.parse_integer(allow_boolean=False, allow_negative=False)
    else:
        parser.parse_punctuation(":")
        value_type = parser.parse_attribute()
        return parser.resolve_operand(value, value_type)


def parse_dynamic_index_without_type(parser: Parser) -> int | UnresolvedOperand:
    """
    Parses an element in an index list, either an index or an ssa value without a type:
    e.g. `42` or `%val`.
    """
    value = parser.parse_optional_unresolved_operand()
    if value is None:
        return parser.parse_integer(allow_boolean=False, allow_negative=False)
    else:
        return value


def parse_dynamic_index_list_with_types(
    parser: Parser,
    dynamic_index: int,
    *,
    delimiter: Parser.Delimiter = Parser.Delimiter.SQUARE,
) -> tuple[Sequence[SSAValue], Sequence[int]]:
    """
    Parses an in index list, composed of a mix of indices and ssa values with types:
    e.g. `[1, 2, 3, %val : i32, 5]`.
    The `dynamic_index` parameter specifies the sentinel value to use to print an ssa
    value instead of a constant.
    """
    mixed_values = parser.parse_comma_separated_list(
        delimiter, lambda: parse_dynamic_index_with_type(parser)
    )

    values: list[SSAValue] = []
    indices: list[int] = []

    for value_or_index in mixed_values:
        if isinstance(value_or_index, int):
            indices.append(value_or_index)
        else:
            indices.append(dynamic_index)
            values.append(value_or_index)

    return values, indices


def parse_dynamic_index_list_without_types(
    parser: Parser,
    dynamic_index: int,
    *,
    delimiter: Parser.Delimiter = Parser.Delimiter.SQUARE,
) -> tuple[Sequence[UnresolvedOperand], Sequence[int]]:
    """
    Parses an in index list, composed of a mix of indices and ssa values without types:
    e.g. `[1, 2, 3, %val, 5]`.
    The `dynamic_index` parameter specifies the sentinel value to use to print an ssa
    value instead of a constant.
    """
    mixed_values = parser.parse_comma_separated_list(
        delimiter, lambda: parse_dynamic_index_without_type(parser)
    )

    values: list[UnresolvedOperand] = []
    indices: list[int] = []

    for value_or_index in mixed_values:
        if isinstance(value_or_index, int):
            indices.append(value_or_index)
        else:
            indices.append(dynamic_index)
            values.append(value_or_index)

    return values, indices
