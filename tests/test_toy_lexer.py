import pytest

from docs.Toy.toy.frontend.lexer import ToyLexer, ToyTokenKind
from xdsl.utils.lexer import Input


@pytest.mark.parametrize(
    "text", ["0", "01", "123456789", "99", "0x0", "0xf", "0x1234", "0xabcdef"]
)
def test_integer_literal(text: str):
    token = ToyLexer(Input(text, "<unknown>")).lex()
    assert token.kind == ToyTokenKind.NUMBER
    assert token.text == text
