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
from xdsl.ir import Dialect, SSAValue
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

    # Test case 6: Mix of integers and SSA values with custom dynamic index
    dynamic_index = -42
    stream = StringIO()
    printer = Printer(stream)
    values = [TestSSAValue(IndexType()), TestSSAValue(IndexType())]
    print_dynamic_index_list(
        printer,
        values,
        [1, dynamic_index, 3, dynamic_index],
        dynamic_index=dynamic_index,
    )
    assert stream.getvalue() == "[1, %0, 3, %1]"


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
    dynamic_index = -42
    test_values = (TestSSAValue(i32), TestSSAValue(index))
    parser = Parser(ctx, "[%0 : i32, 42, %1 : index]")
    parser.ssa_values["0"] = (test_values[0],)
    parser.ssa_values["1"] = (test_values[1],)
    values, indices = parse_dynamic_index_list_with_types(
        parser, dynamic_index=dynamic_index
    )
    assert len(values) == 2
    assert values[0] is test_values[0]
    assert values[1] is test_values[1]
    assert tuple(indices) == (dynamic_index, 42, dynamic_index)


def test_parse_dynamic_index_list_without_types():
    dynamic_index = -42
    parser = Parser(ctx, "[%0, 42, %1]")
    values, indices = parse_dynamic_index_list_without_types(
        parser, dynamic_index=dynamic_index
    )

    assert len(values) == 2
    assert isinstance(values[0], UnresolvedOperand)
    assert isinstance(values[1], UnresolvedOperand)

    assert tuple(indices) == (dynamic_index, 42, dynamic_index)


def test_parse_dynamic_index_list_with_custom_delimiter():
    dynamic_index = -42
    test_values = (TestSSAValue(i32), TestSSAValue(index))
    parser = Parser(ctx, "(%0 : i32, 42, %1 : index)")
    parser.ssa_values["0"] = (test_values[0],)
    parser.ssa_values["1"] = (test_values[1],)
    values, indices = parse_dynamic_index_list_with_types(
        parser, dynamic_index=dynamic_index, delimiter=Parser.Delimiter.PAREN
    )
    assert len(values) == 2
    assert values[0] is test_values[0]
    assert values[1] is test_values[1]
    assert tuple(indices) == (dynamic_index, 42, dynamic_index)


@pytest.mark.parametrize(
    "name,expected_1,expected_2",
    [
        ("dialect.op_name", "dialect", "op_name"),
        ("dialect.op.name", "dialect", "op.name"),
    ],
)
def test_split_name(name: str, expected_1: str, expected_2: str):
    result_1, result_2 = Dialect.split_name(name)
    assert result_1 == expected_1
    assert result_2 == expected_2


def test_split_name_failure():
    with pytest.raises(ValueError) as e:
        Dialect.split_name("test")

    assert e.value.args[0] == ("Invalid operation or attribute name test.")
