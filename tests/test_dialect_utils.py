from io import StringIO

import pytest

from xdsl.context import MLContext
from xdsl.dialects.builtin import DYNAMIC_INDEX, IndexType, IntegerType, i32
from xdsl.dialects.utils import (
    parse_dynamic_index_list_with_types,
    parse_dynamic_index_list_without_types,
    parse_dynamic_index_with_type,
    parse_dynamic_index_without_type,
    print_dynamic_index_list,
)
from xdsl.ir import SSAValue
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.test_value import TestSSAValue

ctx = MLContext()
index = IndexType()


def test_print_dynamic_index_list():
    # Test case 1: Only integers
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(printer, [], [1, 2, 3])
    assert stream.getvalue() == "[1, 2, 3]"

    # Test case 2: Mix of integers and SSA values
    stream = StringIO()
    printer = Printer(stream)
    values = [TestSSAValue(IndexType()), TestSSAValue(IndexType())]
    print_dynamic_index_list(printer, values, [1, DYNAMIC_INDEX, 3, DYNAMIC_INDEX])
    assert stream.getvalue() == "[1, %0, 3, %1]"

    # Test case 3: With value types
    stream = StringIO()
    printer = Printer(stream)
    values = [TestSSAValue(IndexType()), TestSSAValue(IntegerType(32))]
    value_types = (IndexType(), IntegerType(32))
    print_dynamic_index_list(
        printer, values, [DYNAMIC_INDEX, 2, DYNAMIC_INDEX], value_types
    )
    assert stream.getvalue() == "[%0 : index, 2, %1 : i32]"

    # Test case 4: Custom delimiter
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(printer, [], [1, 2, 3], delimiter=Parser.Delimiter.PAREN)
    assert stream.getvalue() == "(1, 2, 3)"

    # Test case 5: Empty list
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(printer, [], [])
    assert stream.getvalue() == "[]"


@pytest.mark.parametrize(
    "delimiter,expected",
    [
        (Parser.Delimiter.SQUARE, "[1, 2, 3]"),
        (Parser.Delimiter.PAREN, "(1, 2, 3)"),
        (Parser.Delimiter.ANGLE, "<1, 2, 3>"),
        (Parser.Delimiter.BRACES, "{1, 2, 3}"),
    ],
)
def test_print_dynamic_index_list_delimiters(
    delimiter: Parser.Delimiter, expected: str
):
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(printer, [], [1, 2, 3], delimiter=delimiter)
    assert stream.getvalue() == expected


def test_parse_dynamic_index_with_type():
    parser = Parser(ctx, "%0 : i32")
    result = parse_dynamic_index_with_type(parser)
    assert isinstance(result, SSAValue)

    parser = Parser(ctx, "42")
    result = parse_dynamic_index_with_type(parser)
    assert isinstance(result, int)
    assert result == 42


def test_parse_dynamic_index_without_type():
    parser = Parser(ctx, "%0")
    result = parse_dynamic_index_without_type(parser)
    assert isinstance(result, UnresolvedOperand)

    parser = Parser(ctx, "42")
    result = parse_dynamic_index_without_type(parser)
    assert isinstance(result, int)
    assert result == 42


def test_parse_dynamic_index_list_with_types():

    parser = Parser(ctx, "[%0 : i32, 42, %1 : index]")
    parser.ssa_values["0"] = (TestSSAValue(i32),)
    parser.ssa_values["1"] = (TestSSAValue(index),)
    values, indices = parse_dynamic_index_list_with_types(parser)
    assert len(values) == 2
    assert len(indices) == 1
    assert isinstance(values[0], SSAValue)
    assert isinstance(values[1], SSAValue)
    assert indices[0] == 42


def test_parse_dynamic_index_list_without_types():
    parser = Parser(ctx, "[%0, 42, %1]")
    values, indices = parse_dynamic_index_list_without_types(parser)
    assert len(values) == 2
    assert len(indices) == 1
    assert isinstance(values[0], UnresolvedOperand)
    assert isinstance(values[1], UnresolvedOperand)
    assert indices[0] == 42


def test_parse_dynamic_index_list_with_custom_delimiter():
    parser = Parser(ctx, "(%0 : i32, 42, %1 : index)")
    values, indices = parse_dynamic_index_list_with_types(
        parser, delimiter=Parser.Delimiter.PAREN
    )
    parser.ssa_values["0"] = (TestSSAValue(i32),)
    parser.ssa_values["1"] = (TestSSAValue(index),)
    assert len(values) == 2
    assert len(indices) == 1
    assert isinstance(values[0], SSAValue)
    assert isinstance(values[1], SSAValue)
    assert indices[0] == 42
