import pytest

from io import StringIO

from xdsl.printer import Printer
from xdsl.ir import MLContext, Attribute, ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import XDSLParser
from xdsl.dialects.builtin import IntAttr, DictionaryAttr, StringAttr, ArrayAttr, Builtin, SymbolRefAttr, SymbolRefAttr


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
