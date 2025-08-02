import pytest

from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import StringLiteral


def _create_string_literal(input_str: str) -> StringLiteral:
    """Create a StringLiteral from a given input string."""
    input_data = Input(input_str, "test.mlir")
    return StringLiteral(0, len(input_str), input_data)


@pytest.mark.parametrize(
    argnames=("string_literal", "expected_bytes"),
    argvalues=[
        (_create_string_literal('"hello world!"'), b"hello world!"),
        (_create_string_literal('"hello \\\\ world!"'), b"hello \\ world!"),
        (_create_string_literal('"hello \t world!\n"'), b"hello \t world!\n"),
        (_create_string_literal('"\x00\x4f and \\00\\4F"'), b"\x00\x4f and \x00\x4f"),
    ],
)
def test_string_litreal_bytes_constents(
    string_literal: StringLiteral, expected_bytes: bytes
):
    """Test that the StringLiteral contains the expected bytes."""
    assert string_literal.bytes_contents == expected_bytes, (
        f"Expected {expected_bytes!r}, got {string_literal.bytes_contents!r}"
    )
    assert string_literal.string_contents == expected_bytes.decode("utf-8"), (
        f"Expected {expected_bytes.decode('utf-8')!r}, got {string_literal.text!r}"
    )
