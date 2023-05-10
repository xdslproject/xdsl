import pytest

from io import StringIO

from xdsl.dialects.builtin import (
    IntAttr,
    DictionaryAttr,
    StringAttr,
    ArrayAttr,
    Builtin,
    SymbolRefAttr,
)
from xdsl.ir import MLContext, Attribute, Region, ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, irdl_op_definition, IRDLOperation
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Token

# pyright: reportPrivateUsage=false


@pytest.mark.parametrize(
    "input,expected",
    [("0, 1, 1", [0, 1, 1]), ("1, 0, 1", [1, 0, 1]), ("1, 1, 0", [1, 1, 0])],
)
def test_int_list_parser(input: str, expected: list[int]):
    ctx = MLContext()
    parser = Parser(ctx, input)

    int_list = parser.parse_list_of(parser.try_parse_integer_literal, "")
    assert [int(span.text) for span in int_list] == expected


@pytest.mark.parametrize(
    "data",
    [
        dict(a=IntAttr(1), b=IntAttr(2), c=IntAttr(3)),
        dict(
            a=StringAttr("hello"),
            b=IntAttr(2),
            c=ArrayAttr([IntAttr(2), StringAttr("world")]),
        ),
        {},
    ],
)
def test_dictionary_attr(data: dict[str, Attribute]):
    attr = DictionaryAttr(data)

    with StringIO() as io:
        Printer(io).print(attr)
        text = io.getvalue()

    ctx = MLContext()
    ctx.register_dialect(Builtin)

    attr1 = Parser(ctx, text).parse_attribute()
    attr2 = Parser(ctx, text).parse_builtin_dict_attr()
    attr3 = Parser(ctx, "attributes " + text).parse_optional_attr_dict_with_keyword()
    assert isinstance(attr1, DictionaryAttr)
    assert isinstance(attr2, DictionaryAttr)
    assert isinstance(attr3, DictionaryAttr)

    assert attr1.data == data
    assert attr2.data == data
    assert attr3.data == data


@pytest.mark.parametrize("text", ["{}", "{a = 1}", "attr {}"])
def test_dictionary_attr_with_keyword_missing(text: str):
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    assert Parser(ctx, text).parse_optional_attr_dict_with_keyword() == None


@pytest.mark.parametrize(
    "text, reserved_names",
    [("{a = 1}", ["a"]), ("{a = 1}", ["b", "a"]), ("{b = 1, a = 1}", ["a"])],
)
def test_dictionary_attr_with_keyword_reserved(text: str, reserved_names: list[str]):
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    with pytest.raises(ParseError):
        Parser(ctx, "attributes" + text).parse_optional_attr_dict_with_keyword(
            reserved_names
        )


@pytest.mark.parametrize(
    "text, expected",
    [("{b = 1}", ["a"]), ("{c = 1}", ["b", "a"]), ("{b = 1, a = 1}", ["c"])],
)
def test_dictionary_attr_with_keyword_not_reserved(text: str, expected: list[str]):
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    res = Parser(ctx, "attributes" + text).parse_optional_attr_dict_with_keyword(
        expected
    )
    assert isinstance(res, DictionaryAttr)


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = "dummy.attr"


def test_parsing():
    """
    Test that the default attribute parser does not try to
    parse attribute arguments without the delimiters.
    """
    ctx = MLContext()
    ctx.register_attr(DummyAttr)

    prog = '#dummy.attr "foo"'
    parser = Parser(ctx, prog)

    r = parser.parse_attribute()
    assert r == DummyAttr()


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("@foo", SymbolRefAttr("foo")),
        ("@foo::@bar", SymbolRefAttr("foo", ["bar"])),
        ("@foo::@bar::@baz", SymbolRefAttr("foo", ["bar", "baz"])),
    ],
)
def test_symref(ref: str, expected: Attribute | None):
    """
    Test that symbol references are correctly parsed.
    """
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, ref)
    parsed_ref = parser.try_parse_ref_attr()

    assert parsed_ref == expected


@irdl_op_definition
class MultiRegionOp(IRDLOperation):
    name = "test.multi_region"
    r1: Region
    r2: Region


def test_parse_multi_region_mlir():
    ctx = MLContext()
    ctx.register_op(MultiRegionOp)

    op_str = """
    "test.multi_region" () ({
    }, {
    }) : () -> ()
    """

    parser = Parser(ctx, op_str)

    op = parser.parse_op()

    assert len(op.regions) == 2


def test_parse_block_name():
    block_str = """
    ^bb0(%name: i32, %100: i32):
    """

    ctx = MLContext()
    parser = Parser(ctx, block_str)
    block = parser.parse_block()

    assert block.args[0].name_hint == "name"
    assert block.args[1].name_hint is None


@pytest.mark.parametrize(
    "delimiter,open_bracket,close_bracket",
    [
        (Parser.Delimiter.NONE, "", ""),
        (Parser.Delimiter.PAREN, "(", ")"),
        (Parser.Delimiter.SQUARE, "[", "]"),
        (Parser.Delimiter.BRACES, "{", "}"),
        (Parser.Delimiter.ANGLE, "<", ">"),
    ],
)
def test_parse_comma_separated_list(
    delimiter: Parser.Delimiter, open_bracket: str, close_bracket: str
):
    input = open_bracket + "2, 4, 5" + close_bracket
    parser = Parser(MLContext(), input)
    res = parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert res == [2, 4, 5]


@pytest.mark.parametrize(
    "delimiter,open_bracket,close_bracket",
    [
        (Parser.Delimiter.PAREN, "(", ")"),
        (Parser.Delimiter.SQUARE, "[", "]"),
        (Parser.Delimiter.BRACES, "{", "}"),
        (Parser.Delimiter.ANGLE, "<", ">"),
    ],
)
def test_parse_comma_separated_list_empty(
    delimiter: Parser.Delimiter, open_bracket: str, close_bracket: str
):
    input = open_bracket + close_bracket
    parser = Parser(MLContext(), input)
    res = parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert res == []


def test_parse_comma_separated_list_none_delimiter_empty():
    parser = Parser(MLContext(), "o")
    with pytest.raises(ParseError):
        parser.parse_comma_separated_list(
            Parser.Delimiter.NONE, parser.parse_integer, " in test"
        )


@pytest.mark.parametrize(
    "delimiter,open_bracket,close_bracket",
    [
        (Parser.Delimiter.PAREN, "(", ")"),
        (Parser.Delimiter.SQUARE, "[", "]"),
        (Parser.Delimiter.BRACES, "{", "}"),
        (Parser.Delimiter.ANGLE, "<", ">"),
    ],
)
def test_parse_comma_separated_list_error_element(
    delimiter: Parser.Delimiter, open_bracket: str, close_bracket: str
):
    input = open_bracket + "o" + close_bracket
    parser = Parser(MLContext(), input)
    with pytest.raises(ParseError) as e:
        parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert e.value.span.text == "o"
    assert e.value.msg == "Expected integer literal"


@pytest.mark.parametrize(
    "delimiter,open_bracket,close_bracket",
    [
        (Parser.Delimiter.PAREN, "(", ")"),
        (Parser.Delimiter.SQUARE, "[", "]"),
        (Parser.Delimiter.BRACES, "{", "}"),
        (Parser.Delimiter.ANGLE, "<", ">"),
    ],
)
def test_parse_comma_separated_list_error_delimiters(
    delimiter: Parser.Delimiter, open_bracket: str, close_bracket: str
):
    input = open_bracket + "2, 4 5"
    parser = Parser(MLContext(), input)
    with pytest.raises(ParseError) as e:
        parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert e.value.span.text == "5"
    assert e.value.msg == "Expected '" + close_bracket + "' in test"


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().values())
)
def test_is_punctuation_true(punctuation: Token.Kind):
    assert punctuation.is_punctuation()


@pytest.mark.parametrize(
    "punctuation", [Token.Kind.BARE_IDENT, Token.Kind.EOF, Token.Kind.INTEGER_LIT]
)
def test_is_punctuation_false(punctuation: Token.Kind):
    assert not punctuation.is_punctuation()


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().values())
)
def test_is_spelling_of_punctuation_true(punctuation: Token.Kind):
    assert Token.Kind.is_spelling_of_punctuation(punctuation.value)


@pytest.mark.parametrize("punctuation", [">-", "o", "4", "$", "_", "@"])
def test_is_spelling_of_punctuation_false(punctuation: str):
    assert not Token.Kind.is_spelling_of_punctuation(punctuation)


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().values())
)
def test_get_punctuation_kind(punctuation: Token.Kind):
    assert (
        punctuation.get_punctuation_kind_from_spelling(punctuation.value) == punctuation
    )


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_punctuation(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), punctuation)

    parser._synchronize_lexer_and_tokenizer()
    res = parser.parse_punctuation(punctuation)
    assert res == punctuation
    parser._synchronize_lexer_and_tokenizer()
    assert parser._parse_token(Token.Kind.EOF, "").kind == Token.Kind.EOF


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_punctuation_fail(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), "e +")
    parser._synchronize_lexer_and_tokenizer()
    with pytest.raises(ParseError) as e:
        parser.parse_punctuation(punctuation, " in test")
    assert e.value.span.text == "e"
    assert e.value.msg == "Expected '" + punctuation + "' in test"


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_optional_punctuation(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), punctuation)
    parser._synchronize_lexer_and_tokenizer()
    res = parser.parse_optional_punctuation(punctuation)
    assert res == punctuation
    parser._synchronize_lexer_and_tokenizer()
    assert parser._parse_token(Token.Kind.EOF, "").kind == Token.Kind.EOF


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_optional_punctuation_fail(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), "e +")
    parser._synchronize_lexer_and_tokenizer()
    assert parser.parse_optional_punctuation(punctuation) is None


@pytest.mark.parametrize(
    "text, expected_value",
    [
        ("true", True),
        ("false", False),
        ("True", None),
        ("False", None),
    ],
)
def test_parse_boolean(text: str, expected_value: bool | None):
    parser = Parser(MLContext(), text)
    assert parser.parse_optional_boolean() == expected_value

    parser = Parser(MLContext(), text)
    if expected_value is None:
        with pytest.raises(ParseError):
            parser.parse_boolean()
    else:
        assert parser.parse_boolean() == expected_value


@pytest.mark.parametrize(
    "text, expected_value, allow_boolean, allow_negative",
    [
        ("42", 42, False, False),
        ("42", 42, True, False),
        ("42", 42, False, True),
        ("42", 42, True, True),
        ("-1", None, False, False),
        ("-1", None, True, False),
        ("-1", -1, False, True),
        ("-1", -1, True, True),
        ("true", None, False, False),
        ("true", 1, True, False),
        ("true", None, False, True),
        ("true", 1, True, True),
        ("false", None, False, False),
        ("false", 0, True, False),
        ("false", None, False, True),
        ("false", 0, True, True),
        ("True", None, True, True),
        ("False", None, True, True),
        ("0x1a", 26, False, False),
        ("0x1a", 26, True, False),
        ("0x1a", 26, False, True),
        ("0x1a", 26, True, True),
        ("-0x1a", None, False, False),
        ("-0x1a", None, True, False),
        ("-0x1a", -26, False, True),
        ("-0x1a", -26, True, True),
    ],
)
def test_parse_int(
    text: str, expected_value: int | None, allow_boolean: bool, allow_negative: bool
):
    parser = Parser(MLContext(), text)
    assert (
        parser.parse_optional_integer(
            allow_boolean=allow_boolean, allow_negative=allow_negative
        )
        == expected_value
    )

    parser = Parser(MLContext(), text)
    if expected_value is None:
        with pytest.raises(ParseError):
            parser.parse_integer(
                allow_boolean=allow_boolean, allow_negative=allow_negative
            )
    else:
        assert (
            parser.parse_integer(
                allow_boolean=allow_boolean, allow_negative=allow_negative
            )
            == expected_value
        )


@pytest.mark.parametrize(
    "text, allow_boolean, allow_negative",
    [
        ("-false", False, True),
        ("-false", True, True),
        ("-true", False, True),
        ("-true", True, True),
        ("-k", True, True),
        ("-(", False, True),
    ],
)
def test_parse_optional_int_error(text: str, allow_boolean: bool, allow_negative: bool):
    """Test that parsing a negative without an integer after raise an error."""
    parser = Parser(MLContext(), text)
    with pytest.raises(ParseError):
        parser.parse_optional_integer(
            allow_boolean=allow_boolean, allow_negative=allow_negative
        )

    parser = Parser(MLContext(), text)
    with pytest.raises(ParseError):
        parser.parse_integer(allow_boolean=allow_boolean, allow_negative=allow_negative)


@pytest.mark.parametrize(
    "text, expected_value",
    [
        ("42", 42),
        ("-1", -1),
        ("true", None),
        ("false", None),
        ("0x1a", 26),
        ("-0x1a", -26),
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
def test_parse_number(text: str, expected_value: int | float | None):
    parser = Parser(MLContext(), text)
    assert parser.parse_optional_number() == expected_value

    parser = Parser(MLContext(), text)
    if expected_value is None:
        with pytest.raises(ParseError):
            parser.parse_number()
    else:
        assert parser.parse_number() == expected_value


@pytest.mark.parametrize(
    "text",
    [
        ("-false"),
        ("-true"),
        ("-k"),
        ("-("),
    ],
)
def test_parse_number_error(text: str):
    """
    Test that parsing a negative without an
    integer or a float after raise an error.
    """
    parser = Parser(MLContext(), text)
    with pytest.raises(ParseError):
        parser.parse_optional_number()

    parser = Parser(MLContext(), text)
    with pytest.raises(ParseError):
        parser.parse_number()
