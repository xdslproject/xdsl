import re
from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import I64, DenseArrayBase, IndexType, IntegerType, i32, i64
from xdsl.dialects.utils import (
    get_dynamic_index_list,
    parse_dynamic_index_list_with_types,
    parse_dynamic_index_list_without_types,
    parse_dynamic_index_with_type,
    parse_dynamic_index_without_type,
    print_dynamic_index_list,
    split_dynamic_index_list,
)
from xdsl.dialects.utils.dynamic_index_list import verify_dynamic_index_list
from xdsl.ir import Dialect, SSAValue
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.test_value import create_ssa_value

ctx = Context()
index = IndexType()

DYNAMIC_INDEX = -42


def test_split_dynamic_index_list():
    # Test case 1: Only integers
    values = [1, 2, 3]
    static, dynamic = split_dynamic_index_list(values, DYNAMIC_INDEX)
    assert static == [1, 2, 3]
    assert dynamic == []

    # Test case 2: Mix of integers and SSA values
    val1 = create_ssa_value(IndexType())
    val2 = create_ssa_value(IndexType())
    values = [1, 2, val1, 4, val2]
    static, dynamic = split_dynamic_index_list(values, DYNAMIC_INDEX)
    assert static == [1, 2, DYNAMIC_INDEX, 4, DYNAMIC_INDEX]
    assert dynamic == [val1, val2]

    # Test case 3: All SSA values
    values = [val1, val2]
    static, dynamic = split_dynamic_index_list(values, DYNAMIC_INDEX)
    assert static == [DYNAMIC_INDEX, DYNAMIC_INDEX]
    assert dynamic == [val1, val2]

    # Test case 4: Empty list
    static, dynamic = split_dynamic_index_list([], DYNAMIC_INDEX)
    assert static == []
    assert dynamic == []


def test_get_dynamic_index_list():
    # Test case 1: Only integers
    static_values = [1, 2, 3]
    result = get_dynamic_index_list(static_values, [], DYNAMIC_INDEX)
    assert result == [1, 2, 3]

    # Test case 2: Mix of integers and SSA values
    val1 = create_ssa_value(IndexType())
    val2 = create_ssa_value(IndexType())
    static_values = [1, 2, DYNAMIC_INDEX, 4, DYNAMIC_INDEX]
    dynamic_values = [val1, val2]
    result = get_dynamic_index_list(static_values, dynamic_values, DYNAMIC_INDEX)
    assert result == [1, 2, val1, 4, val2]

    # Test case 3: All SSA values
    static_values = [DYNAMIC_INDEX, DYNAMIC_INDEX]
    dynamic_values = [val1, val2]
    result = get_dynamic_index_list(static_values, dynamic_values, DYNAMIC_INDEX)
    assert result == [val1, val2]


def test_verify_dynamic_index_list():
    # Test case 1: Valid input
    static_values = [1, 2, DYNAMIC_INDEX]
    dynamic_values = [create_ssa_value(IndexType())]
    verify_dynamic_index_list(static_values, dynamic_values, DYNAMIC_INDEX)

    # Test case 2: Invalid input (mismatched lengths)
    static_values = [1, 2, DYNAMIC_INDEX]
    with pytest.raises(
        VerifyException,
        match=re.escape(
            "The number of dynamic positions passed as values (0) does not match "
            "the number of dynamic position markers (1)."
        ),
    ):
        verify_dynamic_index_list(static_values, [], DYNAMIC_INDEX)


def test_print_dynamic_index_list():
    # Test case 1: Only integers
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(printer, DYNAMIC_INDEX, [], [1, 2, 3])
    assert stream.getvalue() == "[1, 2, 3]"

    # Test case 2: Mix of integers and SSA values
    stream = StringIO()
    printer = Printer(stream)
    values = [create_ssa_value(IndexType()), create_ssa_value(IndexType())]
    print_dynamic_index_list(
        printer,
        DYNAMIC_INDEX,
        values,
        [1, DYNAMIC_INDEX, 3, DYNAMIC_INDEX],
    )
    assert stream.getvalue() == "[1, %0, 3, %1]"

    # Test case 3: With value types
    stream = StringIO()
    printer = Printer(stream)
    values = [create_ssa_value(IndexType()), create_ssa_value(IntegerType(32))]
    value_types = (IndexType(), IntegerType(32))
    print_dynamic_index_list(
        printer,
        DYNAMIC_INDEX,
        values,
        [DYNAMIC_INDEX, 2, DYNAMIC_INDEX],
        value_types,
    )
    assert stream.getvalue() == "[%0 : index, 2, %1 : i32]"

    # Test case 4: Custom delimiter
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(
        printer,
        DYNAMIC_INDEX,
        [],
        [1, 2, 3],
        delimiter=Parser.Delimiter.PAREN,
    )
    assert stream.getvalue() == "(1, 2, 3)"

    # Test case 5: Empty list
    stream = StringIO()
    printer = Printer(stream)
    print_dynamic_index_list(printer, DYNAMIC_INDEX, [], [])
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
    print_dynamic_index_list(printer, DYNAMIC_INDEX, [], [1, 2, 3], delimiter=delimiter)
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
    test_values = (create_ssa_value(i32), create_ssa_value(index))
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
    test_values = (create_ssa_value(i32), create_ssa_value(index))
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
    with pytest.raises(ValueError, match="Invalid operation or attribute name test."):
        Dialect.split_name("test")
