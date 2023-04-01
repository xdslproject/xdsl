from io import StringIO

import pytest

from xdsl.dialects.builtin import (IntAttr, DictionaryAttr, StringAttr,
                                   ArrayAttr, Builtin, SymbolRefAttr)
from xdsl.ir import (MLContext, Attribute, Operation, Region,
                     ParametrizedAttribute)
from xdsl.irdl import irdl_attr_definition, irdl_op_definition
from xdsl.parser import BaseParser, XDSLParser, MLIRParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError


@pytest.mark.parametrize("input,expected", [("0, 1, 1", [0, 1, 1]),
                                            ("1, 0, 1", [1, 0, 1]),
                                            ("1, 1, 0", [1, 1, 0])])
def test_int_list_parser(input: str, expected: list[int]):
    ctx = MLContext()
    parser = XDSLParser(ctx, input)

    int_list = parser.parse_list_of(parser.try_parse_integer_literal, '')
    assert [int(span.text) for span in int_list] == expected


@pytest.mark.parametrize('data', [
    dict(a=IntAttr(1), b=IntAttr(2), c=IntAttr(3)),
    dict(a=StringAttr('hello'),
         b=IntAttr(2),
         c=ArrayAttr([IntAttr(2), StringAttr('world')])),
    dict(),
])
def test_dictionary_attr(data: dict[str, Attribute]):
    attr = DictionaryAttr(data)

    with StringIO() as io:
        Printer(io).print(attr)
        text = io.getvalue()

    ctx = MLContext()
    ctx.register_dialect(Builtin)

    attr = XDSLParser(ctx, text).parse_attribute()
    assert isinstance(attr, DictionaryAttr)

    assert attr.data == data


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = 'dummy.attr'


def test_parsing():
    """
    Test that the default attribute parser does not try to
    parse attribute arguments without the delimiters.
    """
    ctx = MLContext()
    ctx.register_attr(DummyAttr)

    prog = '#dummy.attr "foo"'
    parser = XDSLParser(ctx, prog)

    r = parser.parse_attribute()
    assert r == DummyAttr()


@pytest.mark.parametrize("ref,expected", [
    ("@foo", SymbolRefAttr("foo")),
    ("@foo::@bar", SymbolRefAttr("foo", ["bar"])),
    ("@foo::@bar::@baz", SymbolRefAttr("foo", ["bar", "baz"])),
])
def test_symref(ref: str, expected: Attribute | None):
    """
    Test that symbol references are correctly parsed.
    """
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, ref)
    parsed_ref = parser.try_parse_ref_attr()

    assert parsed_ref == expected


@irdl_op_definition
class MultiRegionOp(Operation):
    name = "test.multi_region"
    r1: Region
    r2: Region


def test_parse_multi_region_mlir():
    ctx = MLContext()
    ctx.register_op(MultiRegionOp)

    op_str = """
    "test.multi_region" () ({
    }, {
    }) : () -> ()
    """

    parser = MLIRParser(ctx, op_str)

    op = parser.parse_op()

    assert len(op.regions) == 2


def test_parse_multi_region_xdsl():
    ctx = MLContext()
    ctx.register_op(MultiRegionOp)

    op_str = """
    "test.multi_region" () {
    } {
    }
    """

    parser = XDSLParser(ctx, op_str)

    op = parser.parse_op()

    assert len(op.regions) == 2


def test_parse_block_name():
    block_str = """
    ^bb0(%name: !i32, %100: !i32):
    """

    ctx = MLContext()
    parser = XDSLParser(ctx, block_str)
    block = parser.parse_block()

    assert block.args[0].name == 'name'
    assert block.args[1].name is None


@pytest.mark.parametrize("delimiter,open_bracket,close_bracket",
                         [(BaseParser.Delimiter.PAREN, '(', ')'),
                          (BaseParser.Delimiter.SQUARE, '[', ']'),
                          (BaseParser.Delimiter.BRACES, '{', '}'),
                          (BaseParser.Delimiter.ANGLE, '<', '>')])
def test_parse_comma_separated_list(delimiter: BaseParser.Delimiter,
                                    open_bracket: str, close_bracket: str):
    input = open_bracket + "2, 4, 5" + close_bracket
    parser = XDSLParser(MLContext(), input)
    res = parser.parse_comma_separated_list(delimiter,
                                            parser.parse_int_literal,
                                            ' in test')
    assert res == [2, 4, 5]


@pytest.mark.parametrize("delimiter,open_bracket,close_bracket",
                         [(BaseParser.Delimiter.PAREN, '(', ')'),
                          (BaseParser.Delimiter.SQUARE, '[', ']'),
                          (BaseParser.Delimiter.BRACES, '{', '}'),
                          (BaseParser.Delimiter.ANGLE, '<', '>')])
def test_parse_comma_separated_list_empty(delimiter: BaseParser.Delimiter,
                                          open_bracket: str,
                                          close_bracket: str):
    input = open_bracket + close_bracket
    parser = XDSLParser(MLContext(), input)
    res = parser.parse_comma_separated_list(delimiter,
                                            parser.parse_int_literal,
                                            ' in test')
    assert res == []


@pytest.mark.parametrize("delimiter,open_bracket,close_bracket",
                         [(BaseParser.Delimiter.PAREN, '(', ')'),
                          (BaseParser.Delimiter.SQUARE, '[', ']'),
                          (BaseParser.Delimiter.BRACES, '{', '}'),
                          (BaseParser.Delimiter.ANGLE, '<', '>')])
def test_parse_comma_separated_list_error_element(
        delimiter: BaseParser.Delimiter, open_bracket: str,
        close_bracket: str):
    input = open_bracket + "o" + close_bracket
    parser = XDSLParser(MLContext(), input)
    with pytest.raises(ParseError) as e:
        parser.parse_comma_separated_list(delimiter, parser.parse_int_literal,
                                          ' in test')
    assert e.value.span.text == 'o'
    assert e.value.msg == "Expected integer literal here"


@pytest.mark.parametrize("delimiter,open_bracket,close_bracket",
                         [(BaseParser.Delimiter.PAREN, '(', ')'),
                          (BaseParser.Delimiter.SQUARE, '[', ']'),
                          (BaseParser.Delimiter.BRACES, '{', '}'),
                          (BaseParser.Delimiter.ANGLE, '<', '>')])
def test_parse_comma_separated_list_error_delimiters(
        delimiter: BaseParser.Delimiter, open_bracket: str,
        close_bracket: str):
    input = open_bracket + "2, 4 5"
    parser = XDSLParser(MLContext(), input)
    with pytest.raises(ParseError) as e:
        parser.parse_comma_separated_list(delimiter, parser.parse_int_literal,
                                          ' in test')
    assert e.value.span.text == '5'
    assert e.value.msg == "Expected '" + close_bracket + "' in test"
