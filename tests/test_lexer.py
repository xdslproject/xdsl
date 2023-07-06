import pytest

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Input, Lexer, Token


def get_token(input: str) -> Token:
    file = Input(input, "<unknown>")
    lexer = Lexer(file)
    token = lexer.lex()
    return token


def assert_single_token(
    input: str, expected_kind: Token.Kind, expected_text: str | None = None
):
    if expected_text is None:
        expected_text = input

    token = get_token(input)

    assert token.kind == expected_kind
    assert token.text == expected_text


def assert_token_fail(input: str):
    file = Input(input, "<unknown>")
    lexer = Lexer(file)
    with pytest.raises(ParseError):
        lexer.lex()


@pytest.mark.parametrize(
    "text,kind",
    [
        ("->", Token.Kind.ARROW),
        (":", Token.Kind.COLON),
        (",", Token.Kind.COMMA),
        ("...", Token.Kind.ELLIPSIS),
        ("=", Token.Kind.EQUAL),
        (">", Token.Kind.GREATER),
        ("{", Token.Kind.L_BRACE),
        ("(", Token.Kind.L_PAREN),
        ("[", Token.Kind.L_SQUARE),
        ("<", Token.Kind.LESS),
        ("-", Token.Kind.MINUS),
        ("+", Token.Kind.PLUS),
        ("?", Token.Kind.QUESTION),
        ("}", Token.Kind.R_BRACE),
        (")", Token.Kind.R_PAREN),
        ("]", Token.Kind.R_SQUARE),
        ("*", Token.Kind.STAR),
        ("|", Token.Kind.VERTICAL_BAR),
        ("{-#", Token.Kind.FILE_METADATA_BEGIN),
        ("#-}", Token.Kind.FILE_METADATA_END),
    ],
)
def test_punctuation(text: str, kind: Token.Kind):
    assert_single_token(text, kind)


@pytest.mark.parametrize("text", [".", "&", "/"])
def test_punctuation_fail(text: str):
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text", ['""', '"@"', '"foo"', '"\\""', '"\\n"', '"\\\\"', '"\\t"']
)
def test_str_literal(text: str):
    assert_single_token(text, Token.Kind.STRING_LIT)


@pytest.mark.parametrize("text", ['"', '"\\"', '"\\a"', '"\n"', '"\v"', '"\f"'])
def test_str_literal_fail(text: str):
    assert_token_fail(text)


@pytest.mark.parametrize(
    "text", ["a", "A", "_", "a_", "a1", "a1_", "a1_2", "a1_2_3", "a$_.", "a$_.1"]
)
def test_bare_ident(text: str):
    """bare-id ::= (letter|[_]) (letter|digit|[_$.])*"""
    assert_single_token(text, Token.Kind.BARE_IDENT)


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
    assert_single_token(text, Token.Kind.AT_IDENT)


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
    assert_single_token("#" + text, Token.Kind.HASH_IDENT)
    assert_single_token("%" + text, Token.Kind.PERCENT_IDENT)
    assert_single_token("^" + text, Token.Kind.CARET_IDENT)
    assert_single_token("!" + text, Token.Kind.EXCLAMATION_IDENT)


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
    assert_single_token("#" + text, Token.Kind.HASH_IDENT, "#" + expected)
    assert_single_token("%" + text, Token.Kind.PERCENT_IDENT, "%" + expected)
    assert_single_token("^" + text, Token.Kind.CARET_IDENT, "^" + expected)
    assert_single_token("!" + text, Token.Kind.EXCLAMATION_IDENT, "!" + expected)


@pytest.mark.parametrize("text", ["0", "01", "123456789", "99", "0x1234", "0xabcdef"])
def test_integer_literal(text: str):
    assert_single_token(text, Token.Kind.INTEGER_LIT)


@pytest.mark.parametrize(
    "text,expected", [("0a", "0"), ("0xg", "0"), ("0xfg", "0xf"), ("0xf.", "0xf")]
)
def test_integer_literal_split(text: str, expected: str):
    assert_single_token(text, Token.Kind.INTEGER_LIT, expected)


@pytest.mark.parametrize(
    "text", ["0.", "1.", "0.2", "38.1243", "92.54e43", "92.5E43", "43.3e-54", "32.E+25"]
)
def test_float_literal(text: str):
    assert_single_token(text, Token.Kind.FLOAT_LIT)


@pytest.mark.parametrize(
    "text,expected", [("3.9e", "3.9"), ("4.5e+", "4.5"), ("5.8e-", "5.8")]
)
def test_float_literal_split(text: str, expected: str):
    assert_single_token(text, Token.Kind.FLOAT_LIT, expected)


@pytest.mark.parametrize("text", ["0", " 0", "   0", "\n0", "\t0", "// Comment\n0"])
def test_whitespace_skip(text: str):
    assert_single_token(text, Token.Kind.INTEGER_LIT, "0")


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "// Comment\n"])
def test_eof(text: str):
    assert_single_token(text, Token.Kind.EOF, "")


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
    assert token.kind == Token.Kind.INTEGER_LIT
    assert token.get_int_value() == expected


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
    assert token.kind == Token.Kind.FLOAT_LIT
    assert token.get_float_value() == expected


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
    assert token.kind == Token.Kind.STRING_LIT
    assert token.get_string_literal_value() == expected
