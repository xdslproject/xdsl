import pytest

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import StringLiteral


@pytest.mark.parametrize(
    argnames=("input_string", "expected_bytes"),
    argvalues=[
        ('"hello world!"', b"hello world!"),
        ('"hello \\\\ world!"', b"hello \\ world!"),
        ('"hello \t world!\n"', b"hello \t world!\n"),
        ('"\x00\x4f and \\00\\4F"', b"\x00\x4f and \x00\x4f"),
    ],
)
def test_string_litreal_bytes_constents(input_string: str, expected_bytes: bytes):
    """Test that the StringLiteral contains the expected bytes."""
    string_literal = StringLiteral(
        0, len(input_string), Input(input_string, "test.mlir")
    )
    assert string_literal.bytes_contents == expected_bytes, (
        f"Expected {expected_bytes!r}, got {string_literal.bytes_contents!r}"
    )
    assert string_literal.string_contents == expected_bytes.decode("utf-8"), (
        f"Expected {expected_bytes.decode('utf-8')!r}, got {string_literal.text!r}"
    )


def test_invalid_escape_in_string_literal():
    """Test that an invalid escape sequence raises a ParseError."""

    input_string = '"hello world!\\"'
    string_literal = StringLiteral(
        0, len(input_string), Input(input_string, "test.mlir")
    )
    with pytest.raises(
        ParseError, match="Incomplete escape sequence at end of string."
    ):
        string_literal.bytes_contents

    input_string = '"hello \\xzz world!"'
    string_literal = StringLiteral(
        0, len(input_string), Input(input_string, "test.mlir")
    )
    with pytest.raises(ParseError, match="Invalid escape sequence: "):
        string_literal.bytes_contents
