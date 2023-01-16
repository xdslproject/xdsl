import pytest

from xdsl.ir import MLContext
from xdsl.parser import Parser


@pytest.mark.parametrize("input,expected", [("0, 1, 1", [0, 1, 1]),
                                            ("1, 0, 1", [1, 0, 1]),
                                            ("1, 1, 0", [1, 1, 0])])
def test_int_list_parser(input: str, expected: list[int]):
    ctx = MLContext()
    parser = Parser(ctx, input)

    int_list = parser.parse_list(parser.parse_int_literal)
    assert int_list == expected


@pytest.mark.parametrize("input,expected", [('"A"=0, "B"=1, "C"=2', {
    "A": 0,
    "B": 1,
    "C": 2
}), ('"MA"=10, "BR"=7, "Z"=3', {
    "MA": 10,
    "BR": 7,
    "Z": 3
}), ('"Q"=77, "VV"=12, "AA"=-8', {
    "Q": 77,
    "VV": 12,
    "AA": -8
})])
def test_int_dictionary_parser(input: str, expected: dict[str, int]):
    ctx = MLContext()
    parser = Parser(ctx, input)

    int_dict = parser.parse_dictionary(parser.parse_optional_str_literal,
                                       parser.parse_optional_int_literal)
    assert int_dict == expected
