import pytest

from xdsl.ir import MLContext
from xdsl.parser import Parser


@pytest.mark.parametrize("input,expected", [("0, 1, 1", [0, 1, 1]),
                                            ("1, 0, 1", [1, 0, 1]),
                                            ("1, 1, 0", [1, 1, 0])])
def test_int_list_parser(input, expected):
    ctx = MLContext()
    parser = Parser(ctx, input)

    int_list = parser.parse_list(parser.parse_int_literal)
    assert int_list == expected
