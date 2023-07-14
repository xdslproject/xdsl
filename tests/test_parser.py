from io import StringIO
from typing import cast

import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
    DictionaryAttr,
    IntAttr,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.dialects.test import Test
from xdsl.ir import Attribute, MLContext, ParametrizedAttribute, Region
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    region_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Token

# pyright: reportPrivateUsage=false


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
    attr2 = Parser(ctx, "attributes " + text).parse_optional_attr_dict_with_keyword()
    assert isinstance(attr1, DictionaryAttr)
    assert isinstance(attr2, DictionaryAttr)

    assert attr1.data == data
    assert attr2.data == data


@pytest.mark.parametrize("text", ["{}", "{a = 1}", "attr {}"])
def test_dictionary_attr_with_keyword_missing(text: str):
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    assert Parser(ctx, text).parse_optional_attr_dict_with_keyword() is None


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
    "text,expected",
    [
        ("@a", StringAttr("a")),
        ("@_", StringAttr("_")),
        ("@a1_2", StringAttr("a1_2")),
        ("@a$_.", StringAttr("a$_.")),
        ('@"foo"', StringAttr("foo")),
        ('@"@"', StringAttr("@")),
        ('@"\\t"', StringAttr("\t")),
        ("f", None),
        ('"f"', None),
    ],
)
def test_symbol_name(text: str, expected: StringAttr | None):
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, text)
    assert parser.parse_optional_symbol_name() == expected

    parser = Parser(ctx, text)
    if expected is not None:
        assert parser.parse_symbol_name() == expected
    else:
        with pytest.raises(ParseError):
            parser.parse_symbol_name()


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("@foo", SymbolRefAttr("foo")),
        ("@foo::@bar", SymbolRefAttr("foo", ["bar"])),
        ("@foo::@bar:", SymbolRefAttr("foo", ["bar"])),
        ('@foo::@"bar"', SymbolRefAttr("foo", ["bar"])),
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
    parsed_ref = parser.parse_attribute()

    assert parsed_ref == expected


@pytest.mark.parametrize(
    "text,expected,expect_type",
    [
        ("%foo", ("foo", None), False),
        ("%-1foo", ("-1foo", None), False),
        ("%foo : i32", ("foo", i32), True),
        ("i32 : %bar", None, False),
        ("i32 : %bar", None, True),
        ("i32 %bar", None, False),
        ("i32 %bar", None, True),
        ("i32", None, False),
        ("i32", None, True),
    ],
)
def test_parse_argument(
    text: str, expected: tuple[str, Attribute | None] | None, expect_type: bool
):
    """
    arg ::= percent-id              if expect_type is False
    arg ::= percent-id ':' type     if expect_type is True
    """
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    # parse_optional_argument
    parser = Parser(ctx, text)
    res = parser.parse_optional_argument(expect_type)
    if expected is not None:
        assert res is not None
        assert res.name.text[1:] == expected[0]
        assert res.type == expected[1]
    else:
        assert res is None

    # parse_argument
    parser = Parser(ctx, text)
    if expected is not None:
        res = parser.parse_argument(expect_type)
        assert res is not None
        assert res.name.text[1:] == expected[0]
        assert res.type == expected[1]
    else:
        with pytest.raises(ParseError):
            parser.parse_argument(expect_type)


@pytest.mark.parametrize(
    "text,expect_type",
    [
        ("%foo : %bar", True),
        ("%foo : ", True),
        ("%foo", True),
        ("%foo %bar", True),
    ],
)
def test_parse_argument_fail(text: str, expect_type: bool):
    ctx = MLContext()
    ctx.register_dialect(Builtin)

    # parse_optional_argument
    parser = Parser(ctx, text)
    with pytest.raises(ParseError):
        parser.parse_optional_argument(expect_type)

    # parse_argument
    parser = Parser(ctx, text)
    with pytest.raises(ParseError):
        parser.parse_argument(expect_type)


@pytest.mark.parametrize(
    "text,num_ops_and_args",
    [
        # no blocks
        ("{}", []),
        # One entry block
        ("""{ "test.op"() : () -> () }""", [(1, 0)]),
        # One entry block and another block
        (
            """{
          "test.op"() : () -> ()
        ^bb0(%x: i32):
          "test.op"() : () -> ()
          "test.op"() : () -> ()
        }""",
            [(1, 0), (2, 1)],
        ),
        # One labeled entry block and another block
        (
            """{
        ^bb0:
          "test.op"() : () -> ()
          "test.op"() : () -> ()
        ^bb1:
          "test.op"() : () -> ()
        }""",
            [(2, 0), (1, 0)],
        ),
        # One labeled entry block with args and another block
        (
            """{
        ^bb0(%x: i32, %y: i32):
          "test.op"() : () -> ()
          "test.op"() : () -> ()
        ^bb1(%z: i32):
          "test.op"() : () -> ()
        }""",
            [(2, 2), (1, 1)],
        ),
        # Not regions
        ("""^bb:""", None),
        ("""}""", None),
        (""""test.op"() : () -> ()""", None),
    ],
)
def test_parse_region_no_args(
    text: str, num_ops_and_args: list[tuple[int, int]] | None
):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Test)

    parser = Parser(ctx, text)
    if num_ops_and_args is None:
        with pytest.raises(ParseError):
            parser.parse_region()
    else:
        res = parser.parse_region()
        for block, (n_ops, n_args) in zip(res.blocks, num_ops_and_args):
            assert len(block.ops) == n_ops
            assert len(block.args) == n_args

    parser = Parser(ctx, text)
    res = parser.parse_optional_region()
    if num_ops_and_args is None:
        assert res is None
    else:
        assert res is not None
        for block, (n_ops, n_args) in zip(res.blocks, num_ops_and_args):
            assert len(block.ops) == n_ops
            assert len(block.args) == n_args


@pytest.mark.parametrize(
    "text",
    [
        """{""",
        """{ "test.op"() : () -> ()""",
    ],
)
def test_parse_region_fail(text: str):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Test)

    parser = Parser(ctx, text)
    with pytest.raises(ParseError):
        parser.parse_region()

    parser = Parser(ctx, text)
    with pytest.raises(ParseError):
        parser.parse_optional_region()


@pytest.mark.parametrize(
    "text",
    [
        """%x : i32 { "test.op"(%x) : (i32) -> () }""",
        """%x : i32 { "test.op"(%x) : (i32) -> () ^bb0: }""",
    ],
)
def test_parse_region_with_args(text: str):
    """Parse a region with args already provided."""
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Test)

    parser = Parser(ctx, text)
    arg = parser.parse_argument()
    region = parser.parse_region((arg,))
    assert len(region.blocks[0].args) == 1

    parser = Parser(ctx, text)
    arg = parser.parse_argument()
    region = parser.parse_optional_region((arg,))
    assert region is not None
    assert len(region.blocks[0].args) == 1


@pytest.mark.parametrize(
    "text",
    [
        """%x : i32 { ^bb: "test.op"(%x) : (i32) -> () }""",
        """%x : i32 { ^bb(%y : i32): "test.op"(%x) : (i32) -> () }""",
        """%x : i32 { %x = "test.op"() : () -> (i32) }""",
    ],
)
def test_parse_region_with_args_fail(text: str):
    """Parse a region with args already provided."""
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Test)

    parser = Parser(ctx, text)
    arg = parser.parse_argument()
    with pytest.raises(ParseError):
        parser.parse_region((arg,))

    parser = Parser(ctx, text)
    arg = parser.parse_argument()
    with pytest.raises(ParseError):
        parser.parse_optional_region((arg,))


@irdl_op_definition
class MultiRegionOp(IRDLOperation):
    name = "test.multi_region"
    r1: Region = region_def()
    r2: Region = region_def()


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
    block = parser._parse_block()  # pyright: ignore[reportPrivateUsage]

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

    parser = Parser(MLContext(), input)
    if delimiter is Parser.Delimiter.NONE:
        res = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_integer, parser.parse_integer
        )
    else:
        res = parser.parse_optional_comma_separated_list(
            delimiter, parser.parse_integer, " in test"
        )
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
    "delimiter",
    [
        (Parser.Delimiter.PAREN),
        (Parser.Delimiter.SQUARE),
        (Parser.Delimiter.BRACES),
        (Parser.Delimiter.ANGLE),
    ],
)
def test_parse_optional_comma_separated_list(delimiter: Parser.Delimiter):
    parser = Parser(MLContext(), "o")
    res = parser.parse_optional_comma_separated_list(delimiter, parser.parse_integer)
    assert res is None


def test_parse_optional_undelimited_comma_separated_list_empty():
    parser = Parser(MLContext(), "o")
    res = parser.parse_optional_undelimited_comma_separated_list(
        parser.parse_optional_integer, parser.parse_integer
    )
    assert res is None


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
    with pytest.raises(ParseError, match="Expected integer literal"):
        parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")

    parser = Parser(MLContext(), input)
    with pytest.raises(ParseError, match="Expected integer literal"):
        parser.parse_optional_comma_separated_list(
            delimiter, parser.parse_integer, " in test"
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
def test_parse_comma_separated_list_error_delimiters(
    delimiter: Parser.Delimiter, open_bracket: str, close_bracket: str
):
    input = open_bracket + "2, 4 5"
    parser = Parser(MLContext(), input)
    with pytest.raises(ParseError) as e:
        parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert e.value.span.text == "5"
    assert e.value.msg == "Expected '" + close_bracket + "' in test"

    parser = Parser(MLContext(), input)
    with pytest.raises(ParseError) as e:
        parser.parse_optional_comma_separated_list(
            delimiter, parser.parse_integer, " in test"
        )
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
    value = cast(Token.PunctuationSpelling, punctuation.value)
    assert Token.Kind.is_spelling_of_punctuation(value)


@pytest.mark.parametrize("punctuation", [">-", "o", "4", "$", "_", "@"])
def test_is_spelling_of_punctuation_false(punctuation: str):
    assert not Token.Kind.is_spelling_of_punctuation(punctuation)


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().values())
)
def test_get_punctuation_kind(punctuation: Token.Kind):
    value = cast(Token.PunctuationSpelling, punctuation.value)
    assert punctuation.get_punctuation_kind_from_spelling(value) == punctuation


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_punctuation(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), punctuation)

    res = parser.parse_punctuation(punctuation)
    assert res == punctuation
    assert parser._parse_token(Token.Kind.EOF, "").kind == Token.Kind.EOF


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_punctuation_fail(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), "e +")
    with pytest.raises(ParseError) as e:
        parser.parse_punctuation(punctuation, " in test")
    assert e.value.span.text == "e"
    assert e.value.msg == "Expected '" + punctuation + "' in test"


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_optional_punctuation(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), punctuation)
    res = parser.parse_optional_punctuation(punctuation)
    assert res == punctuation
    assert parser._parse_token(Token.Kind.EOF, "").kind == Token.Kind.EOF


@pytest.mark.parametrize(
    "punctuation", list(Token.Kind.get_punctuation_spelling_to_kind_dict().keys())
)
def test_parse_optional_punctuation_fail(punctuation: Token.PunctuationSpelling):
    parser = Parser(MLContext(), "e +")
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
