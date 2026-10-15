import pytest

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Input, Location, Position, Span
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRToken, MLIRTokenKind


def get_token(input: bytes) -> MLIRToken:
    file = Input(input, "<unknown>")
    lexer = MLIRLexer(file)
    token = lexer.lex()
    return token


def assert_single_token(
    input: bytes, expected_kind: MLIRTokenKind, expected_text: bytes | None = None
):
    if expected_text is None:
        expected_text = input

    token = get_token(input)

    assert token.kind == expected_kind
    assert token.span.bytes == expected_text


def assert_token_fail(input: bytes):
    file = Input(input, "<unknown>")
    lexer = MLIRLexer(file)
    with pytest.raises(ParseError):
        lexer.lex()


def test_input_str_deprecated():
    with pytest.warns(DeprecationWarning, match="Passing str to Input"):
        input = Input("é", "test.mlir")
    assert input.content == b"\xc3\xa9"


def test_byte_offsets():
    input = Input('"é😀" x\r\ny'.encode(), "test.mlir")
    assert len(input) == 13
    assert input.at(1) == 0xC3
    assert input.at(len(input)) is None
    assert input.slice(1, 3) == b"\xc3\xa9"
    assert input.slice(0, len(input) + 1) is None

    lexer = MLIRLexer(input)
    literal = lexer.lex()
    assert literal.text == '"é😀"'
    assert literal.span.bytes == '"é😀"'.encode()
    assert (literal.span.start, literal.span.end) == (0, 8)
    assert literal.kind.get_string_literal_value(literal.span) == "é😀"
    token = lexer.lex()
    assert token.text == "x"
    assert token.span.get_location() == Location("test.mlir", 1, 10)
    token = lexer.lex()
    assert token.text == "y"
    assert token.span.start == 12
    assert token.span.get_location() == Location("test.mlir", 2, 1)


@pytest.mark.parametrize(
    "source,start,end,expected",
    [
        ('"é" x'.encode(), 5, 6, 'test.mlir:1:6\n"é" x\n    ^\n'),
        ('"é"'.encode(), 1, 2, 'test.mlir:1:2\n"é"\n ^\n'),
        (b"\xff x", 2, 3, "test.mlir:1:3\n� x\n  ^\n"),
        ('"é"\r\nx'.encode(), 1, 7, 'test.mlir:1:2\n"é"\n ^^\nx\n^\n'),
        ("é\n".encode(), 3, 3, "test.mlir:2:1\n\n^\n"),
        (b"abc\nxyz", 0, 4, "test.mlir:1:1\nabc\n^^^\n"),
    ],
)
def test_byte_span_diagnostic(source: bytes, start: int, end: int, expected: str):
    span = Span(start, end, Input(source, "test.mlir"))
    assert span.print_with_context() == expected
    # A debug representation must also work for partial UTF-8 sequences.
    assert repr(span) == f"Span[{start}:{end}]({source[start:end]!r})"


@pytest.mark.parametrize(
    "text,kind",
    [
        (b"->", MLIRTokenKind.ARROW),
        (b":", MLIRTokenKind.COLON),
        (b",", MLIRTokenKind.COMMA),
        (b"...", MLIRTokenKind.ELLIPSIS),
        (b"=", MLIRTokenKind.EQUAL),
        (b">", MLIRTokenKind.GREATER),
        (b"{", MLIRTokenKind.L_BRACE),
        (b"(", MLIRTokenKind.L_PAREN),
        (b"[", MLIRTokenKind.L_SQUARE),
        (b"<", MLIRTokenKind.LESS),
        (b"-", MLIRTokenKind.MINUS),
        (b"+", MLIRTokenKind.PLUS),
        (b"?", MLIRTokenKind.QUESTION),
        (b"}", MLIRTokenKind.R_BRACE),
        (b")", MLIRTokenKind.R_PAREN),
        (b"]", MLIRTokenKind.R_SQUARE),
        (b"/", MLIRTokenKind.SLASH),
        (b"*", MLIRTokenKind.STAR),
        (b"|", MLIRTokenKind.VERTICAL_BAR),
        (b"{-#", MLIRTokenKind.FILE_METADATA_BEGIN),
        (b"#-}", MLIRTokenKind.FILE_METADATA_END),
    ],
)
def test_punctuation(text: bytes, kind: MLIRTokenKind):
    assert_single_token(text, kind)


@pytest.mark.parametrize("text", [b".", b"&"])
def test_punctuation_fail(text: bytes):
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text", [b'""', b'"@"', b'"foo"', b'"\\""', b'"\\n"', b'"\\\\"', b'"\\t"']
)
def test_str_literal(text: bytes):
    assert_single_token(text, MLIRTokenKind.STRING_LIT)


@pytest.mark.parametrize("text", [b'"', b'"\\"', b'"\\a"', b'"\n"', b'"\v"', b'"\f"'])
def test_str_literal_fail(text: bytes):
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text",
    [b"a", b"A", b"_", b"a_", b"a1", b"a1_", b"a1_2", b"a1_2_3", b"a$_.", b"a$_.1"],
)
def test_bare_ident(text: bytes):
    """bare-id ::= (letter|[_]) (letter|digit|[_$.])*"""
    assert_single_token(text, MLIRTokenKind.BARE_IDENT)


@pytest.mark.parametrize(
    "text",
    [
        b"@a",
        b"@A",
        b"@_",
        b"@a_",
        b"@a1",
        b"@a1_",
        b"@a1_2",
        b"@a1_2_3",
        b"@a$_.",
        b"@a$_.1",
        b'@""',
        b'@"@"',
        b'@"foo"',
        b'@"\\""',
        b'@"\\n"',
        b'@"\\\\"',
        b'@"\\t"',
    ],
)
def test_at_ident(text: bytes):
    """at-ident ::= `@` (bare-id | string-literal)"""
    assert_single_token(text, MLIRTokenKind.AT_IDENT)


@pytest.mark.parametrize(
    "text",
    [
        b"@",
        b'@"',
        b'@"\\"',
        b'@"\\a"',
        b'@"\n"',
        b'@"\v"',
        b'@"\f"',
        b'@ "a"',
        b"@ f",
        b"@$",
    ],
)
def test_at_ident_fail(text: bytes):
    """at-ident ::= `@` (bare-id | string-literal)"""
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text",
    [b"0", b"1234", b"a", b"S", b"$", b"_", b".", b"-", b"e_.$-324", b"e5$-e_", b"foo"],
)
def test_prefixed_ident(text: bytes):
    """hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    """percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    """caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    """exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    assert_single_token(b"#" + text, MLIRTokenKind.HASH_IDENT)
    assert_single_token(b"%" + text, MLIRTokenKind.PERCENT_IDENT)
    assert_single_token(b"^" + text, MLIRTokenKind.CARET_IDENT)
    assert_single_token(b"!" + text, MLIRTokenKind.EXCLAMATION_IDENT)


@pytest.mark.parametrize("text", [b"+", b'""', b"#", b"%", b"^", b"!", b"\n", b""])
def test_prefixed_ident_fail(text: bytes):
    """
    hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    """
    assert_token_fail(b"#" + text)
    assert_token_fail(b"%" + text)
    assert_token_fail(b"^" + text)
    assert_token_fail(b"!" + text)


@pytest.mark.parametrize(
    "text,expected",
    [
        (b"0x0", b"0"),
        (b"0e", b"0"),
        (b"0$", b"0"),
        (b"0_", b"0"),
        (b"0-", b"0"),
        (b"0.", b"0"),
    ],
)
def test_prefixed_ident_split(text: bytes, expected: bytes):
    """Check that the prefixed identifier is split at the right character."""
    assert_single_token(b"#" + text, MLIRTokenKind.HASH_IDENT, b"#" + expected)
    assert_single_token(b"%" + text, MLIRTokenKind.PERCENT_IDENT, b"%" + expected)
    assert_single_token(b"^" + text, MLIRTokenKind.CARET_IDENT, b"^" + expected)
    assert_single_token(b"!" + text, MLIRTokenKind.EXCLAMATION_IDENT, b"!" + expected)


@pytest.mark.parametrize(
    "text", [b"0", b"01", b"123456789", b"99", b"0x1234", b"0xabcdef"]
)
def test_integer_literal(text: bytes):
    assert_single_token(text, MLIRTokenKind.INTEGER_LIT)


@pytest.mark.parametrize(
    "text,expected",
    [(b"0a", b"0"), (b"0xg", b"0"), (b"0xfg", b"0xf"), (b"0xf.", b"0xf")],
)
def test_integer_literal_split(text: bytes, expected: bytes):
    assert_single_token(text, MLIRTokenKind.INTEGER_LIT, expected)


@pytest.mark.parametrize(
    "text",
    [
        b"0.",
        b"1.",
        b"0.2",
        b"38.1243",
        b"92.54e43",
        b"92.5E43",
        b"43.3e-54",
        b"32.E+25",
    ],
)
def test_float_literal(text: bytes):
    assert_single_token(text, MLIRTokenKind.FLOAT_LIT)


@pytest.mark.parametrize(
    "text,expected", [(b"3.9e", b"3.9"), (b"4.5e+", b"4.5"), (b"5.8e-", b"5.8")]
)
def test_float_literal_split(text: bytes, expected: bytes):
    assert_single_token(text, MLIRTokenKind.FLOAT_LIT, expected)


def test_slash_vs_comment():
    """A single `/` produces SLASH; `//` is consumed as a comment."""
    assert_single_token(b"/", MLIRTokenKind.SLASH)

    file = Input(b"// this is a comment containing invalid UTF-8: \xff\n0", "<unknown>")
    lexer = MLIRLexer(file)
    token = lexer.lex()
    assert token.kind == MLIRTokenKind.INTEGER_LIT
    assert token.text == "0"


@pytest.mark.parametrize(
    "text", [b"0", b" 0", b"   0", b"\n0", b"\t0", b"// Comment\n0"]
)
def test_whitespace_skip(text: bytes):
    assert_single_token(text, MLIRTokenKind.INTEGER_LIT, b"0")


@pytest.mark.parametrize("text", [b"", b"   ", b"\n\n", b"// Comment\n"])
def test_eof(text: bytes):
    assert_single_token(text, MLIRTokenKind.EOF, b"")


@pytest.mark.parametrize(
    "text, expected",
    [
        (b"0", 0),
        (b"010", 10),
        (b"123456789", 123456789),
        (b"0x1234", 4660),
        (b"0xabcdef23", 2882400035),
    ],
)
def test_token_get_int_value(text: bytes, expected: int):
    token = get_token(text)
    assert token.kind == MLIRTokenKind.INTEGER_LIT
    assert token.kind.get_int_value(token.span) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (b"0.", 0.0),
        (b"1.", 1.0),
        (b"0.2", 0.2),
        (b"38.1243", 38.1243),
        (b"92.54e43", 92.54e43),
        (b"92.5E43", 92.5e43),
        (b"43.3e-54", 43.3e-54),
        (b"32.E+25", 32.0e25),
    ],
)
def test_token_get_float_value(text: bytes, expected: float):
    token = get_token(text)
    assert token.kind == MLIRTokenKind.FLOAT_LIT
    assert token.kind.get_float_value(token.span) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (b'""', ""),
        (b'"@"', "@"),
        (b'"foo"', "foo"),
        (b'"\\""', '"'),
        (b'"\\n"', "\n"),
        (b'"\\\\"', "\\"),
        (b'"\\t"', "\t"),
    ],
)
def test_token_get_string_literal_value(text: bytes, expected: float):
    token = get_token(text)
    assert token.kind == MLIRTokenKind.STRING_LIT
    assert token.kind.get_string_literal_value(token.span) == expected


START_END_LINE_CONTENT = b"""\
0123
5678"""


@pytest.mark.parametrize(
    "pos,expected_start,expected_end",
    [
        (0, 0, 4),
        (1, 0, 4),
        (5, 5, 9),
        (9, 5, 9),
    ],
)
def test_get_start_of_line(
    pos: Position, expected_start: Position, expected_end: Position
):
    input = Input(START_END_LINE_CONTENT, "<>")
    assert input.get_start_of_line(pos) == expected_start
    assert input.get_end_of_line(pos) == expected_end
