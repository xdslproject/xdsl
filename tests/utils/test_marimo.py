import pytest

from xdsl.ir.affine import AffineExpr
from xdsl.utils.marimo import Expression


@pytest.mark.parametrize(
    "text, symbols",
    [
        ("", set[str]()),
        ("hello", {"hello"}),
        ("hello + world", {"hello", "world"}),
        ("a + b + a", {"a", "b"}),
    ],
)
def test_parse_symbols(text: str, symbols: set[str]):
    assert Expression.parse_symbols(text) == symbols


@pytest.mark.parametrize(
    "text, expression",
    [
        ("1", Expression([], AffineExpr.constant(1))),
        ("1 + 2", Expression([], AffineExpr.constant(1) + AffineExpr.constant(2))),
        # ("hello", {"hello"}),
        # ("hello + world", {"hello", "world"}),
        # ("a + b + a", {"a", "b"}),
    ],
)
def test_parse_expression(text: str, expression: Expression):
    assert Expression.parse(text) == expression
