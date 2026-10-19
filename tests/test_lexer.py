from enum import Enum, auto

import pytest
from typing_extensions import assert_type

from xdsl.utils.exceptions import MultipleSpansParseError, ParseError
from xdsl.utils.lexer import (
    ContentT,
    Input,
    Lexer,
    Location,
    Offset,
    Position,
    Span,
    Token,
)
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRToken, MLIRTokenKind


@pytest.mark.parametrize(
    "source,size,x_pos,first,prefix",
    [
        ("é😀x\r\ny", 6, 2, "é", "é😀"),
        ("é😀x\r\ny".encode(), 10, 6, 0xC3, "é😀".encode()),
    ],
)
def test_input_representation(
    source: ContentT,
    size: Offset,
    x_pos: Position,
    first: str | int,
    prefix: ContentT,
):
    input = Input(source, "test")
    assert input.content is source
    assert len(input) == size
    assert input.at(0) == first
    assert input.at(size) is None
    assert input.slice(0, x_pos) == prefix
    assert input.slice(-1, size) is None
    assert input.slice(0, size + 1) is None
    assert input.get_start_of_line(x_pos) == 0
    assert input.get_end_of_line(x_pos) == size - 2
    assert input.get_start_of_line(size - 1) == size - 1
    assert input.get_end_of_line(size - 1) == size
    assert input.get_start_of_line(size) == size - 1
    assert input.get_end_of_line(size) == size


@pytest.mark.parametrize(
    "span,msg,expected",
    [
        pytest.param(
            Span(0, 0, Input("", "test")),
            "error",
            """\
test:1:1

^
error
""",
            id="empty-input",
        ),
        pytest.param(
            Span(4, 4, Input("abc\n", "test")),
            "error",
            """\
test:2:1

^
error
""",
            id="eof-after-newline",
        ),
        pytest.param(
            Span(4, 4, Input("abc\nxyz", "test")),
            "error",
            """\
test:2:1
xyz
^
error
""",
            id="empty-span-at-start-of-nonempty-line",
        ),
        pytest.param(
            Span(0, 4, Input("abc\nxyz", "test")),
            "error",
            """\
test:1:1
abc
^^^
error
""",
            id="exclusive-end-at-next-line",
        ),
        pytest.param(
            Span(1, 5, Input("abc\ndef\nghi", "test")),
            "error",
            """\
test:1:2
abc
 ^^
 error
def
^
""",
            id="multiline",
        ),
        pytest.param(
            Span(1, 6, Input("abc\r\ndef", "test")),
            "error",
            """\
test:1:2
abc
 ^^
 error
def
^
""",
            id="crlf-counts-as-two-characters",
        ),
        pytest.param(
            Span(0, 4, Input("a\n\nb", "test")),
            "error",
            """\
test:1:1
a
^
error

^
b
^
""",
            id="empty-line-inside-span",
        ),
        pytest.param(
            Span(0, 3, Input("a\u2028b", "test")),
            None,
            """\
test:1:1
a\u2028b
^^^
""",
            id="only-lf-splits-source-lines",
        ),
        pytest.param(
            Span(2, 3, Input("\néx", "test"), line_offset=3),
            None,
            """\
test:5:2
éx
 ^
""",
            id="unicode-and-line-offset-without-message",
        ),
        pytest.param(
            Span(3, 3, Input("abc", "test")),
            "error",
            """\
test:1:4
abc
   ^
   error
""",
            id="eof-without-newline",
        ),
    ],
)
def test_span_print_with_context(span: Span, msg: str | None, expected: str):
    assert span.print_with_context(msg) == expected


@pytest.mark.parametrize(
    "source,x_pos,x",
    [("é😀x\r\ny", 2, "x"), ("é😀x\r\ny".encode(), 6, b"x")],
)
def test_span_representation(source: ContentT, x_pos: Position, x: ContentT):
    input = Input(source, "test")
    size = len(input)
    span = Span(x_pos, x_pos + 1, input)
    assert span.content == x
    assert span.text == "x"
    assert span.bytes == b"x"
    assert len(span) == 1
    assert span.get_location() == Location("test", 1, x_pos + 1)
    assert Span(size - 1, size, input, line_offset=4).get_location() == Location(
        "test", 6, 1
    )


def test_span_accessor_types():
    text_span = Span(1, 2, Input("é😀x", "test"))
    byte_span = Span(2, 6, Input("é😀x".encode(), "test"))
    assert_type(text_span, Span[str])
    assert_type(byte_span, Span[bytes])
    assert assert_type(text_span.content, str) == "😀"
    assert assert_type(byte_span.content, bytes) == "😀".encode()
    assert assert_type(text_span.text, str) == "😀"
    assert assert_type(byte_span.text, str) == "😀"
    assert assert_type(text_span.bytes, bytes) == "😀".encode()
    assert assert_type(byte_span.bytes, bytes) == "😀".encode()
    assert len(text_span) == 1
    assert len(byte_span) == 4
    assert text_span.get_location() == Location("test", 1, 2)
    assert byte_span.get_location() == Location("test", 1, 3)


@pytest.mark.parametrize("source", [b"\xff", b"\xc3", b"\xa9"])
def test_span_text_strict_utf8(source: bytes):
    span = Span(1, 2, Input(b"x" + source + b"y", "test"))
    assert span.content == source
    assert span.bytes == source
    with pytest.raises(UnicodeDecodeError):
        _ = span.text
    assert repr(span) == f"Span[1:2]({source!r})"


def test_span_bytes_strict_utf8():
    span = Span(1, 2, Input("x\ud800y", "test"))
    assert span.content == "\ud800"
    assert span.text == "\ud800"
    with pytest.raises(UnicodeEncodeError):
        _ = span.bytes
    assert repr(span) == "Span[1:2]('\\ud800')"


@pytest.mark.parametrize("source", ["'\"\n\\", b"'\"\n\\", "", b""])
def test_span_repr(source: ContentT):
    span = Span(0, len(source), Input(source, "test"))
    assert repr(span) == f"Span[0:{len(source)}]({source!r})"


@pytest.mark.parametrize("use_bytes", [False, True])
@pytest.mark.parametrize("msg", [None, "error"])
@pytest.mark.parametrize(
    "source,start,end,expected",
    [
        ("", 0, 0, "test:1:1\n\n^\n"),
        ("abc", 3, 3, "test:1:4\nabc\n   ^\n"),
        ("é\n", 2, 2, "test:2:1\n\n^\n"),
        ("abc\nxyz", 0, 4, "test:1:1\nabc\n^^^\n"),
        ("abc\ndef\nghi", 1, 5, "test:1:2\nabc\n ^^\ndef\n^\n"),
        ('"é"\r\nx', 1, 6, 'test:1:2\n"é"\n ^^\nx\n^\n'),
        ("abc\r\nxyz", 0, 5, "test:1:1\nabc\n^^^\n"),
        ("a\n\nb", 0, 4, "test:1:1\na\n^\n\n^\nb\n^\n"),
        ("abc\nxyz", 3, 4, "test:1:4\nabc\n   ^\n"),
        ("a\vb\fc\u2028d", 0, 7, "test:1:1\na\vb\fc\u2028d\n^^^^^^^\n"),
    ],
)
def test_span_diagnostic(
    source: str,
    start: Position,
    end: Position,
    expected: str,
    use_bytes: bool,
    msg: str | None,
):
    span = (
        Span(
            len(source[:start].encode()),
            len(source[:end].encode()),
            Input(source.encode(), "test"),
        )
        if use_bytes
        else Span(start, end, Input(source, "test"))
    )
    if msg is not None:
        # Place the message beneath the first caret line, with the same indent.
        lines = expected.split("\n")
        indent = lines[2].index("^")
        lines.insert(3, " " * indent + msg)
        expected = "\n".join(lines)
    assert span.print_with_context(msg) == expected


class UnitTokenKind(Enum):
    UNIT = auto()
    EOF = auto()


class UnitLexer(Lexer[UnitTokenKind, ContentT]):
    """A test lexer that returns one source element per token."""

    def lex(self) -> Token[UnitTokenKind, ContentT]:
        start = self.pos
        if start == len(self.input):
            return self._form_token(UnitTokenKind.EOF, start)
        self.pos += 1
        return self._form_token(UnitTokenKind.UNIT, start)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("é😀x", ("é", "😀", "x")),
        (
            "é😀x".encode() + b"\xff",
            (b"\xc3", b"\xa9", b"\xf0", b"\x9f", b"\x98", b"\x80", b"x", b"\xff"),
        ),
    ],
)
def test_lexer_representation(source: ContentT, expected: tuple[ContentT, ...]):
    input = Input(source, "test")
    lexer = UnitLexer(input)
    for i, text in enumerate(expected):
        token = lexer.lex()
        assert token.kind == UnitTokenKind.UNIT
        assert token.content == text
        assert token.span == Span(i, i + 1, input)
    eof = lexer.lex()
    assert eof.kind == UnitTokenKind.EOF
    assert eof.span == Span(len(expected), len(expected), input)
    assert not eof.content


def test_token_accessor_types():
    text_token = Token(UnitTokenKind.UNIT, Span(1, 2, Input("é😀x", "test")))
    byte_token = Token(UnitTokenKind.UNIT, Span(2, 6, Input("é😀x".encode(), "test")))
    assert_type(text_token, Token[UnitTokenKind, str])
    assert_type(byte_token, Token[UnitTokenKind, bytes])
    assert assert_type(text_token.content, str) == "😀"
    assert assert_type(byte_token.content, bytes) == "😀".encode()
    assert assert_type(text_token.text, str) == "😀"
    assert assert_type(byte_token.text, str) == "😀"
    assert assert_type(text_token.bytes, bytes) == "😀".encode()
    assert assert_type(byte_token.bytes, bytes) == "😀".encode()


@pytest.mark.parametrize(
    "source,start,end,expected",
    [
        ('"é😀" x'.encode(), 9, 10, 'test:1:10\n"é😀" x\n     ^\n     error\n'),
        ('"é"'.encode(), 1, 2, 'test:1:2\n"é"\n ^\n error\n'),
        (b"\xff x", 2, 3, "test:1:3\n� x\n  ^\n  error\n"),
        ('"é"\r\nx'.encode(), 1, 7, 'test:1:2\n"é"\n ^^\n error\nx\n^\n'),
        ("é\n".encode(), 3, 3, "test:2:1\n\n^\nerror\n"),
        (b"abc\nxyz", 0, 4, "test:1:1\nabc\n^^^\nerror\n"),
        (b"", 0, 0, "test:1:1\n\n^\nerror\n"),
    ],
)
def test_byte_span_diagnostic(
    source: bytes, start: Position, end: Position, expected: str
):
    span = Span(start, end, Input(source, "test"))
    assert str(ParseError(span, "error")) == expected
    # Debugging a span must not attempt to decode partial or malformed UTF-8.
    assert repr(source[start:end]) in repr(span)


def test_mixed_span_diagnostic():
    error = MultipleSpansParseError(
        Span(2, 3, Input("é x".encode(), "bytes"), line_offset=2),
        "error",
        "reference:",
        [(Span(2, 3, Input("é x", "text")), "defined here")],
    )
    assert str(error) == (
        "bytes:3:3\né x\n ^\n error\nreference:\ntext:1:3\né x\n  ^\n  defined here\n\n"
    )


def get_token(input: str) -> MLIRToken:
    file = Input(input, "<unknown>")
    lexer = MLIRLexer(file)
    token = lexer.lex()
    return token


def assert_single_token(
    input: str, expected_kind: MLIRTokenKind, expected_text: str | None = None
):
    if expected_text is None:
        expected_text = input

    token = get_token(input)

    assert token.kind == expected_kind
    assert token.text == expected_text


def assert_token_fail(input: str):
    file = Input(input, "<unknown>")
    lexer = MLIRLexer(file)
    with pytest.raises(ParseError):
        lexer.lex()


@pytest.mark.parametrize(
    "text,kind",
    [
        ("->", MLIRTokenKind.ARROW),
        (":", MLIRTokenKind.COLON),
        (",", MLIRTokenKind.COMMA),
        ("...", MLIRTokenKind.ELLIPSIS),
        ("=", MLIRTokenKind.EQUAL),
        (">", MLIRTokenKind.GREATER),
        ("{", MLIRTokenKind.L_BRACE),
        ("(", MLIRTokenKind.L_PAREN),
        ("[", MLIRTokenKind.L_SQUARE),
        ("<", MLIRTokenKind.LESS),
        ("-", MLIRTokenKind.MINUS),
        ("+", MLIRTokenKind.PLUS),
        ("?", MLIRTokenKind.QUESTION),
        ("}", MLIRTokenKind.R_BRACE),
        (")", MLIRTokenKind.R_PAREN),
        ("]", MLIRTokenKind.R_SQUARE),
        ("/", MLIRTokenKind.SLASH),
        ("*", MLIRTokenKind.STAR),
        ("|", MLIRTokenKind.VERTICAL_BAR),
        ("{-#", MLIRTokenKind.FILE_METADATA_BEGIN),
        ("#-}", MLIRTokenKind.FILE_METADATA_END),
    ],
)
def test_punctuation(text: str, kind: MLIRTokenKind):
    assert_single_token(text, kind)


@pytest.mark.parametrize("text", [".", "&"])
def test_punctuation_fail(text: str):
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text", ['""', '"@"', '"foo"', '"\\""', '"\\n"', '"\\\\"', '"\\t"']
)
def test_str_literal(text: str):
    assert_single_token(text, MLIRTokenKind.STRING_LIT)


@pytest.mark.parametrize("text", ['"', '"\\"', '"\\a"', '"\n"', '"\v"', '"\f"'])
def test_str_literal_fail(text: str):
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text", ["a", "A", "_", "a_", "a1", "a1_", "a1_2", "a1_2_3", "a$_.", "a$_.1"]
)
def test_bare_ident(text: str):
    """bare-id ::= (letter|[_]) (letter|digit|[_$.])*"""
    assert_single_token(text, MLIRTokenKind.BARE_IDENT)


@pytest.mark.parametrize(
    "text",
    [
        "@a",
        "@A",
        "@_",
        "@a_",
        "@a1",
        "@a1_",
        "@a1_2",
        "@a1_2_3",
        "@a$_.",
        "@a$_.1",
        '@""',
        '@"@"',
        '@"foo"',
        '@"\\""',
        '@"\\n"',
        '@"\\\\"',
        '@"\\t"',
    ],
)
def test_at_ident(text: str):
    """at-ident ::= `@` (bare-id | string-literal)"""
    assert_single_token(text, MLIRTokenKind.AT_IDENT)


@pytest.mark.parametrize(
    "text",
    ["@", '@"', '@"\\"', '@"\\a"', '@"\n"', '@"\v"', '@"\f"', '@ "a"', "@ f", "@$"],
)
def test_at_ident_fail(text: str):
    """at-ident ::= `@` (bare-id | string-literal)"""
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text", ["0", "1234", "a", "S", "$", "_", ".", "-", "e_.$-324", "e5$-e_", "foo"]
)
def test_prefixed_ident(text: str):
    """hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    """percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    """caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    """exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)"""
    assert_single_token("#" + text, MLIRTokenKind.HASH_IDENT)
    assert_single_token("%" + text, MLIRTokenKind.PERCENT_IDENT)
    assert_single_token("^" + text, MLIRTokenKind.CARET_IDENT)
    assert_single_token("!" + text, MLIRTokenKind.EXCLAMATION_IDENT)


@pytest.mark.parametrize("text", ["+", '""', "#", "%", "^", "!", "\n", ""])
def test_prefixed_ident_fail(text: str):
    """
    hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)
    """
    assert_token_fail("#" + text)
    assert_token_fail("%" + text)
    assert_token_fail("^" + text)
    assert_token_fail("!" + text)


@pytest.mark.parametrize(
    "text,expected",
    [("0x0", "0"), ("0e", "0"), ("0$", "0"), ("0_", "0"), ("0-", "0"), ("0.", "0")],
)
def test_prefixed_ident_split(text: str, expected: str):
    """Check that the prefixed identifier is split at the right character."""
    assert_single_token("#" + text, MLIRTokenKind.HASH_IDENT, "#" + expected)
    assert_single_token("%" + text, MLIRTokenKind.PERCENT_IDENT, "%" + expected)
    assert_single_token("^" + text, MLIRTokenKind.CARET_IDENT, "^" + expected)
    assert_single_token("!" + text, MLIRTokenKind.EXCLAMATION_IDENT, "!" + expected)


@pytest.mark.parametrize(
    "text", ["0", "01", "123456789", "99", "0x0", "0xf", "0x1234", "0xabcdef"]
)
def test_integer_literal(text: str):
    assert_single_token(text, MLIRTokenKind.INTEGER_LIT)


@pytest.mark.parametrize(
    "text,expected", [("0a", "0"), ("0xg", "0"), ("0xfg", "0xf"), ("0xf.", "0xf")]
)
def test_integer_literal_split(text: str, expected: str):
    assert_single_token(text, MLIRTokenKind.INTEGER_LIT, expected)


@pytest.mark.parametrize(
    "text", ["0.", "1.", "0.2", "38.1243", "92.54e43", "92.5E43", "43.3e-54", "32.E+25"]
)
def test_float_literal(text: str):
    assert_single_token(text, MLIRTokenKind.FLOAT_LIT)


@pytest.mark.parametrize(
    "text,expected", [("3.9e", "3.9"), ("4.5e+", "4.5"), ("5.8e-", "5.8")]
)
def test_float_literal_split(text: str, expected: str):
    assert_single_token(text, MLIRTokenKind.FLOAT_LIT, expected)


def test_slash_vs_comment():
    """A single `/` produces SLASH; `//` is consumed as a comment."""
    assert_single_token("/", MLIRTokenKind.SLASH)

    file = Input("// this is a comment\n0", "<unknown>")
    lexer = MLIRLexer(file)
    token = lexer.lex()
    assert token.kind == MLIRTokenKind.INTEGER_LIT
    assert token.text == "0"


@pytest.mark.parametrize("text", ["0", " 0", "   0", "\n0", "\t0", "// Comment\n0"])
def test_whitespace_skip(text: str):
    assert_single_token(text, MLIRTokenKind.INTEGER_LIT, "0")


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "// Comment\n"])
def test_eof(text: str):
    assert_single_token(text, MLIRTokenKind.EOF, "")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0),
        ("010", 10),
        ("123456789", 123456789),
        ("0x1234", 4660),
        ("0xabcdef23", 2882400035),
    ],
)
def test_token_get_int_value(text: str, expected: int):
    token = get_token(text)
    assert token.kind == MLIRTokenKind.INTEGER_LIT
    assert token.kind.get_int_value(token.span) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.", 0.0),
        ("1.", 1.0),
        ("0.2", 0.2),
        ("38.1243", 38.1243),
        ("92.54e43", 92.54e43),
        ("92.5E43", 92.5e43),
        ("43.3e-54", 43.3e-54),
        ("32.E+25", 32.0e25),
    ],
)
def test_token_get_float_value(text: str, expected: float):
    token = get_token(text)
    assert token.kind == MLIRTokenKind.FLOAT_LIT
    assert token.kind.get_float_value(token.span) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ('""', ""),
        ('"@"', "@"),
        ('"foo"', "foo"),
        ('"\\""', '"'),
        ('"\\n"', "\n"),
        ('"\\\\"', "\\"),
        ('"\\t"', "\t"),
    ],
)
def test_token_get_string_literal_value(text: str, expected: float):
    token = get_token(text)
    assert token.kind == MLIRTokenKind.STRING_LIT
    assert token.kind.get_string_literal_value(token.span) == expected


START_END_LINE_CONTENT = """\
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
