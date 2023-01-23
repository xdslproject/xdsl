from io import StringIO

import pytest

from xdsl.printer import Printer
from xdsl.ir import MLContext, Attribute
from xdsl.parser import XDSLParser
from xdsl.dialects.builtin import IntAttr, DictionaryAttr, StringAttr, ArrayAttr, Builtin


@pytest.mark.parametrize("input,expected", [("0, 1, 1", [0, 1, 1]),
                                            ("1, 0, 1", [1, 0, 1]),
                                            ("1, 1, 0", [1, 1, 0])])
def test_int_list_parser(input: str, expected: list[int]):
    ctx = MLContext()
    parser = XDSLParser(ctx, input)

    int_list = parser.parse_list_of(parser.try_parse_integer_literal, '')
    assert [int(span.text) for span in int_list] == expected


@pytest.mark.parametrize('data', [
    dict(a=IntAttr.from_int(1), b=IntAttr.from_int(2), c=IntAttr.from_int(3)),
    dict(a=StringAttr.from_str('hello'),
         b=IntAttr.from_int(2),
         c=ArrayAttr.from_list(
             [IntAttr.from_int(2),
              StringAttr.from_str('world')])),
    dict(),
])
def test_dictionary_attr(data: dict[str, Attribute]):
    attr = DictionaryAttr.from_dict(data)

    with StringIO() as io:
        Printer(io).print(attr)
        text = io.getvalue()

    ctx = MLContext()
    ctx.register_dialect(Builtin)

    attr = XDSLParser(ctx, text).parse_attribute()

    assert attr.data == data
