import builtins
import re
from io import StringIO
from typing import cast

import pytest

from xdsl.context import Context
from xdsl.dialect_interfaces.op_asm import OpAsmDialectInterface
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ArrayAttr,
    Builtin,
    DictionaryAttr,
    FileLineColLoc,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    UnknownLoc,
    i32,
)
from xdsl.dialects.test import Test
from xdsl.ir import Attribute, Block, ParametrizedAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    region_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError
from xdsl.utils.mlir_lexer import (
    KIND_BY_PUNCTUATION_SPELLING,
    MLIRTokenKind,
    PunctuationSpelling,
)
from xdsl.utils.str_enum import StrEnum

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
        Printer(io).print_attribute(attr)
        text = io.getvalue()

    ctx = Context()
    ctx.load_dialect(Builtin)

    attr1 = Parser(ctx, text).parse_attribute()
    attr2 = Parser(ctx, "attributes " + text).parse_optional_attr_dict_with_keyword()
    assert isinstance(attr1, DictionaryAttr)
    assert isinstance(attr2, DictionaryAttr)

    assert attr1.data == data
    assert attr2.data == data


@pytest.mark.parametrize("text", ["{}", "{a = 1}", "attr {}"])
def test_dictionary_attr_with_keyword_missing(text: str):
    ctx = Context()
    ctx.load_dialect(Builtin)

    assert Parser(ctx, text).parse_optional_attr_dict_with_keyword() is None


@pytest.mark.parametrize(
    "text, reserved_names",
    [("{a = 1}", ["a"]), ("{a = 1}", ["b", "a"]), ("{b = 1, a = 1}", ["a"])],
)
def test_dictionary_attr_with_keyword_reserved(text: str, reserved_names: list[str]):
    ctx = Context()
    ctx.load_dialect(Builtin)

    with pytest.raises(ParseError):
        Parser(ctx, "attributes" + text).parse_optional_attr_dict_with_keyword(
            reserved_names
        )


@pytest.mark.parametrize(
    "text, expected",
    [("{b = 1}", ["a"]), ("{c = 1}", ["b", "a"]), ("{b = 1, a = 1}", ["c"])],
)
def test_dictionary_attr_with_keyword_not_reserved(text: str, expected: list[str]):
    ctx = Context()
    ctx.load_dialect(Builtin)

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
    ctx = Context()
    ctx.load_attr_or_type(DummyAttr)

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
    ctx = Context()
    ctx.load_dialect(Builtin)

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
    ctx = Context()
    ctx.load_dialect(Builtin)

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
    ctx = Context()
    ctx.load_dialect(Builtin)

    # parse_optional_argument
    parser = Parser(ctx, text)
    res = parser.parse_optional_argument(expect_type)
    if expected is not None:
        assert res is not None
        assert res.name.text[1:] == expected[0]
        if expected[1] is not None:
            assert isinstance(res, Parser.Argument)
            assert res.type == expected[1]
        else:
            assert isinstance(res, parser.UnresolvedArgument)
    else:
        assert res is None

    # parse_argument
    parser = Parser(ctx, text)
    if expected is not None:
        res = parser.parse_argument(expect_type=expect_type)
        assert res is not None
        assert res.name.text[1:] == expected[0]
        if expected[1] is not None:
            assert isinstance(res, Parser.Argument)
            assert res.type == expected[1]
        else:
            assert isinstance(res, parser.UnresolvedArgument)
    else:
        with pytest.raises(ParseError):
            parser.parse_argument(expect_type=expect_type)


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
    ctx = Context()
    ctx.load_dialect(Builtin)

    # parse_optional_argument
    parser = Parser(ctx, text)
    with pytest.raises(ParseError):
        parser.parse_optional_argument(expect_type)

    # parse_argument
    parser = Parser(ctx, text)
    with pytest.raises(ParseError):
        parser.parse_argument(expect_type=expect_type)


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
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

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
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

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
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

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
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

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
    r1 = region_def()
    r2 = region_def()


def test_parse_multi_region_mlir():
    ctx = Context()
    ctx.load_op(MultiRegionOp)

    op_str = """
    "test.multi_region" () ({
    }, {
    }) : () -> ()
    """

    parser = Parser(ctx, op_str)

    op = parser.parse_op()

    assert len(op.regions) == 2


def test_is_default_block_name():
    assert Block.is_default_block_name("bb0")
    assert Block.is_default_block_name("bb1")
    assert not Block.is_default_block_name("bb1a")
    assert not Block.is_default_block_name("bb")
    assert not Block.is_default_block_name("bbb0")
    assert not Block.is_default_block_name("")


def test_parse_block_name():
    block_str = """
    ^bb0(%name: i32, %100: i32):
    """

    ctx = Context()
    parser = Parser(ctx, block_str)
    block = parser._parse_block()

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

    parser = Parser(Context(), input)
    res = parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert res == [2, 4, 5]

    parser = Parser(Context(), input)
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
    parser = Parser(Context(), input)
    res = parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert res == []


def test_parse_comma_separated_list_none_delimiter_empty():
    parser = Parser(Context(), "o")
    with pytest.raises(ParseError):
        parser.parse_comma_separated_list(
            Parser.Delimiter.NONE, parser.parse_integer, " in test"
        )


def test_parse_comma_separated_list_none_delimiter_two_no_comma():
    """Test that a list without commas will only parse the first element."""
    parser = Parser(Context(), "1 2")
    res = parser.parse_comma_separated_list(
        Parser.Delimiter.NONE, parser.parse_integer, " in test"
    )
    assert res == [1]
    assert parser.parse_optional_integer() is not None

    parser = Parser(Context(), "1 2")
    parser.parse_optional_undelimited_comma_separated_list(
        parser.parse_optional_integer, parser.parse_integer
    )
    assert res == [1]
    assert parser.parse_optional_integer() is not None


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
    parser = Parser(Context(), "o")
    res = parser.parse_optional_comma_separated_list(delimiter, parser.parse_integer)
    assert res is None


def test_parse_optional_undelimited_comma_separated_list_empty():
    parser = Parser(Context(), "o")
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
    parser = Parser(Context(), input)
    with pytest.raises(ParseError, match="Expected integer literal"):
        parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")

    parser = Parser(Context(), input)
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
    parser = Parser(Context(), input)
    with pytest.raises(
        ParseError, match=re.escape(f"'{close_bracket}' expected in test")
    ) as e:
        parser.parse_comma_separated_list(delimiter, parser.parse_integer, " in test")
    assert e.value.span.text == "5"

    parser = Parser(Context(), input)
    with pytest.raises(
        ParseError, match=re.escape(f"'{close_bracket}' expected in test")
    ) as e:
        parser.parse_optional_comma_separated_list(
            delimiter, parser.parse_integer, " in test"
        )
    assert e.value.span.text == "5"


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.values()))
def test_is_punctuation_true(punctuation: MLIRTokenKind):
    assert punctuation.is_punctuation()


@pytest.mark.parametrize(
    "punctuation",
    [MLIRTokenKind.BARE_IDENT, MLIRTokenKind.EOF, MLIRTokenKind.INTEGER_LIT],
)
def test_is_punctuation_false(punctuation: MLIRTokenKind):
    assert not punctuation.is_punctuation()


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.values()))
def test_is_spelling_of_punctuation_true(punctuation: MLIRTokenKind):
    value = cast(PunctuationSpelling, punctuation.value)
    assert MLIRTokenKind.is_spelling_of_punctuation(value)


@pytest.mark.parametrize("punctuation", [">-", "o", "4", "$", "_", "@"])
def test_is_spelling_of_punctuation_false(punctuation: str):
    assert not MLIRTokenKind.is_spelling_of_punctuation(punctuation)


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.values()))
def test_get_punctuation_kind(punctuation: MLIRTokenKind):
    value = cast(PunctuationSpelling, punctuation.value)
    assert punctuation.get_punctuation_kind_from_name(value) == punctuation


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.keys()))
def test_parse_punctuation(punctuation: PunctuationSpelling):
    parser = Parser(Context(), punctuation)

    res = parser.parse_punctuation(punctuation)
    assert res == punctuation
    assert parser._parse_token(MLIRTokenKind.EOF, "").kind == MLIRTokenKind.EOF


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.keys()))
def test_parse_punctuation_fail(punctuation: PunctuationSpelling):
    parser = Parser(Context(), "e +")
    with pytest.raises(ParseError) as e:
        parser.parse_punctuation(punctuation, " in test")
    assert e.value.span.text == "e"
    assert e.value.msg == "Expected '" + punctuation + "' in test"


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.keys()))
def test_parse_optional_punctuation(punctuation: PunctuationSpelling):
    parser = Parser(Context(), punctuation)
    res = parser.parse_optional_punctuation(punctuation)
    assert res == punctuation
    assert parser._parse_token(MLIRTokenKind.EOF, "").kind == MLIRTokenKind.EOF


@pytest.mark.parametrize("punctuation", list(KIND_BY_PUNCTUATION_SPELLING.keys()))
def test_parse_optional_punctuation_fail(punctuation: PunctuationSpelling):
    parser = Parser(Context(), "e +")
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
    parser = Parser(Context(), text)
    assert parser.parse_optional_boolean() == expected_value

    parser = Parser(Context(), text)
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
    parser = Parser(Context(), text)
    assert (
        parser.parse_optional_integer(
            allow_boolean=allow_boolean, allow_negative=allow_negative
        )
        == expected_value
    )

    parser = Parser(Context(), text)
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


@pytest.mark.parametrize("nonnumeric", ["--", "+", "a", "{", "(1.0, 1.0)"])
def test_parse_optional_bool_int_or_float_nonnumeric(nonnumeric: str):
    parser = Parser(Context(), nonnumeric)
    assert parser._parse_optional_bool_int_or_float() is None


@pytest.mark.parametrize(
    "numeric,typ",
    [
        ("1", int),
        ("true", bool),
        ("1.0", float),
        ("-1", int),
        ("-1.0", float),
        ("false", bool),
    ],
)
def test_parse_optional_bool_int_or_float_numeric(numeric: str, typ: type):
    parser = Parser(Context(), numeric)
    value_span = parser._parse_optional_bool_int_or_float()
    assert value_span is not None
    value, span = value_span
    match typ:
        case builtins.int:
            assert value == typ(numeric)
            assert span.text == numeric
        case builtins.float:
            assert value == typ(numeric)
            assert span.text == numeric
        case builtins.bool:
            expected = numeric == "true"
            assert value == expected
            assert span.text == numeric
        case _:
            pytest.fail("unreachable")


@pytest.mark.parametrize("nonnumeric", ["--", "+", "a", "{", "(1.0, 1.0)"])
def test_parse_bool_int_or_float_nonnumeric(nonnumeric: str):
    parser = Parser(Context(), nonnumeric)
    with pytest.raises(ParseError):
        parser._parse_bool_int_or_float()


@pytest.mark.parametrize(
    "numeric,typ",
    [
        ("1", int),
        ("true", bool),
        ("1.0", float),
        ("-1", int),
        ("-1.0", float),
        ("false", bool),
    ],
)
def test_parse_bool_int_or_float_numeric(numeric: str, typ: type):
    parser = Parser(Context(), numeric)
    value_span = parser._parse_optional_bool_int_or_float()
    assert value_span is not None
    value, span = value_span
    match typ:
        case builtins.int:
            assert value == typ(numeric)
            assert span.text == numeric
        case builtins.float:
            assert value == typ(numeric)
            assert span.text == numeric
        case builtins.bool:
            expected = numeric == "true"
            assert value == expected
            assert span.text == numeric
        case _:
            pytest.fail("unreachable")


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
    parser = Parser(Context(), text)
    with pytest.raises(ParseError):
        parser.parse_optional_integer(
            allow_boolean=allow_boolean, allow_negative=allow_negative
        )

    parser = Parser(Context(), text)
    with pytest.raises(ParseError):
        parser.parse_integer(allow_boolean=allow_boolean, allow_negative=allow_negative)


@pytest.mark.parametrize(
    "text, allow_boolean, expected_value",
    [
        ("42", False, 42),
        ("-1", False, -1),
        ("true", False, None),
        ("false", False, None),
        ("0x1a", False, 26),
        ("-0x1a", False, -26),
        ("0.", False, 0.0),
        ("1.", False, 1.0),
        ("0.2", False, 0.2),
        ("38.1243", False, 38.1243),
        ("92.54e43", False, 92.54e43),
        ("92.5E43", False, 92.5e43),
        ("43.3e-54", False, 43.3e-54),
        ("32.E+25", False, 32.0e25),
        ("true", True, 1),
        ("false", True, 0),
        ("42", True, 42),
        ("0.2", True, 0.2),
    ],
)
def test_parse_number(
    text: str, allow_boolean: bool, expected_value: int | float | None
):
    parser = Parser(Context(), text)
    assert parser.parse_optional_number(allow_boolean=allow_boolean) == expected_value

    parser = Parser(Context(), text)
    if expected_value is None:
        with pytest.raises(ParseError):
            parser.parse_number()
    else:
        assert parser.parse_number(allow_boolean=allow_boolean) == expected_value


@pytest.mark.parametrize(
    "text, expected_value",
    [
        ("3: i16", IntegerAttr(3, 16)),
        ("24: i32", IntegerAttr(24, 32)),
        ("0: index", IntegerAttr.from_index_int_value(0)),
        ("-64: i64", IntegerAttr(-64, 64)),
        ("-64.4: f64", FloatAttr(-64.4, 64)),
        ("32.4: f32", FloatAttr(32.4, 32)),
        ("0x7e00 : f16", FloatAttr(float("nan"), 16)),
        ("0x7c00 : f16", FloatAttr(float("inf"), 16)),
        ("0xfc00 : f16", FloatAttr(float("-inf"), 16)),
        ("0x7fc00000 : f32", FloatAttr(float("nan"), 32)),
        ("0x7f800000 : f32", FloatAttr(float("inf"), 32)),
        ("0xff800000 : f32", FloatAttr(float("-inf"), 32)),
        ("0x7ff8000000000000 : f64", FloatAttr(float("nan"), 64)),
        ("0x7ff0000000000000 : f64", FloatAttr(float("inf"), 64)),
        ("0xfff0000000000000 : f64", FloatAttr(float("-inf"), 64)),
        # ("3 : f64", None),  # todo this fails in mlir-opt but not in xdsl
    ],
)
def test_parse_optional_builtin_int_or_float_attr(
    text: str, expected_value: IntegerAttr | FloatAttr
):
    parser = Parser(Context(), text)
    assert parser.parse_optional_builtin_int_or_float_attr() == expected_value


@pytest.mark.parametrize(
    "text, allow_boolean",
    [
        ("-false", False),
        ("-true", False),
        ("-false", True),
        ("-true", True),
        ("-k", False),
        ("-(", False),
    ],
)
def test_parse_number_error(text: str, allow_boolean: bool):
    """
    Test that parsing a negative without an
    integer or a float after raise an error.
    """
    parser = Parser(Context(), text)
    with pytest.raises(ParseError):
        parser.parse_optional_number(allow_boolean=allow_boolean)

    parser = Parser(Context(), text)
    with pytest.raises(ParseError):
        parser.parse_number(allow_boolean=allow_boolean)


@irdl_op_definition
class PropertyOp(IRDLOperation):
    name = "test.prop_op"

    first = prop_def(StringAttr)
    second = prop_def(IntegerAttr[IntegerType])


def test_properties_retrocompatibility():
    # Straightforward case
    ctx = Context()
    ctx.load_op(PropertyOp)
    parser = Parser(ctx, '"test.prop_op"() <{first = "str", second = 42}> : () -> ()')

    op = parser.parse_op()
    assert isinstance(op, PropertyOp)
    op.verify()

    # Retrocompatibility case, only target
    parser = Parser(ctx, '"test.prop_op"() {first = "str", second = 42} : () -> ()')
    retro_op = parser.parse_op()
    assert isinstance(retro_op, PropertyOp)
    retro_op.verify()

    assert op.attributes == retro_op.attributes
    assert op.properties == retro_op.properties

    # Test partial case
    parser = Parser(ctx, '"test.prop_op"() <{first = "str"}> {second = 42} : () -> ()')
    partial_op = parser.parse_op()
    assert isinstance(partial_op, PropertyOp)
    partial_op.verify()

    assert op.attributes == partial_op.attributes
    assert op.properties == partial_op.properties


def test_parse_location():
    ctx = Context()
    attr = Parser(ctx, "loc(unknown)").parse_optional_location()
    assert attr == UnknownLoc()

    attr = Parser(ctx, 'loc("one":2:3)').parse_optional_location()
    assert attr == FileLineColLoc(StringAttr("one"), IntAttr(2), IntAttr(3))

    with pytest.raises(ParseError, match="Unexpected location syntax."):
        Parser(ctx, "loc(unexpected)").parse_optional_location()


@pytest.mark.parametrize(
    "keyword,expected",
    [
        ("public", StringAttr("public")),
        ("nested", StringAttr("nested")),
        ("private", StringAttr("private")),
        ("privateeee", None),
        ("unknown", None),
    ],
)
def test_parse_visibility(keyword: str, expected: StringAttr | None):
    assert Parser(Context(), keyword).parse_optional_visibility_keyword() == expected

    parser = Parser(Context(), keyword)
    if expected is None:
        with pytest.raises(ParseError, match="expect symbol visibility keyword"):
            parser.parse_visibility_keyword()
    else:
        assert parser.parse_visibility_keyword() == expected


class MyEnum(StrEnum):
    A = "a"
    B = "b"
    C = "c"
    D = "d-non-keyword"


@pytest.mark.parametrize(
    "keyword, expected",
    [
        ("a", MyEnum.A),
        ('"a"', MyEnum.A),
        ("b", MyEnum.B),
        ('"b"', MyEnum.B),
        ("c", MyEnum.C),
        ('"c"', MyEnum.C),
        ('"d-non-keyword"', MyEnum.D),
        ("other", None),
        ('"other"', None),
    ],
)
def test_parse_str_enum_right_token(keyword: str, expected: MyEnum | None):
    """
    Test parsing of string enums where the next
    token is a keyword or a string literal.
    """
    if expected is None:
        with pytest.raises(
            ParseError, match="Expected `a`, `b`, `c`, or `d-non-keyword`"
        ):
            Parser(Context(), keyword).parse_optional_str_enum(MyEnum)
    else:
        assert Parser(Context(), keyword).parse_optional_str_enum(MyEnum) == expected

    parser = Parser(Context(), keyword)
    if expected is None:
        with pytest.raises(
            ParseError, match="Expected `a`, `b`, `c`, or `d-non-keyword`"
        ):
            parser.parse_str_enum(MyEnum)
    else:
        assert parser.parse_str_enum(MyEnum) == expected


@pytest.mark.parametrize("keyword", ["2", "-"])
def test_parse_str_enum_wrong_token(keyword: str):
    """
    Test parsing of string enums where the next
    token is a keyword or a string literal.
    """
    assert Parser(Context(), keyword).parse_optional_str_enum(MyEnum) is None

    with pytest.raises(ParseError, match="Expected `a`, `b`, `c`, or `d-non-keyword`"):
        Parser(Context(), keyword).parse_str_enum(MyEnum)


@pytest.mark.parametrize("value", ["(1., 2)", "(1, 2.)"])
def test_parse_optional_complex_error(value: str):
    parser = Parser(Context(), value)
    with pytest.raises(
        ParseError,
        match=re.escape("Complex value must be either (float, float) or (int, int)"),
    ):
        parser._parse_optional_complex()


@pytest.mark.parametrize("noncomplex", ["1", "-1", "A", "{"])
def test_parse_optional_complex_noncomplex(noncomplex: str):
    parser = Parser(Context(), noncomplex)
    assert parser._parse_optional_complex() is None


@pytest.mark.parametrize(
    "toks, pyval", [("(-1., 2.)", (-1.0, 2.0)), ("(1, 2)", (1, 2))]
)
def test_parse_optional_complex_success(
    toks: str, pyval: tuple[int, int] | tuple[float, float]
):
    parser = Parser(Context(), toks)
    value_and_span = parser._parse_optional_complex()
    assert value_and_span is not None
    value, span = value_and_span
    assert value == pyval
    assert span.text == toks


@pytest.mark.parametrize(
    "start, end, text",
    [
        ("<", ">", "<1>"),
        ("[", "]", "[1]"),
        ("{", "}", "{1}"),
        ("(", ")", "(1)"),
        ("{-#", "#-}", "{-# 1 #-}"),
    ],
)
def test_delimiters(start: str, end: str, text: str):
    parser = Parser(Context(), text)
    with parser.delimited(start, end):
        value = parser.parse_integer()

    assert value == 1
    assert parser.pos == len(text)


def test_angle_brackets():
    parser = Parser(Context(), "<1>")
    with parser.in_angle_brackets():
        value = parser.parse_integer()

    assert value == 1
    assert parser.pos == 3


def test_square_brackets():
    parser = Parser(Context(), "[1]")
    with parser.in_square_brackets():
        value = parser.parse_integer()

    assert value == 1
    assert parser.pos == 3


def test_braces():
    parser = Parser(Context(), "{1}")
    with parser.in_braces():
        value = parser.parse_integer()

    assert value == 1
    assert parser.pos == 3


def test_parens():
    parser = Parser(Context(), "(1)")
    with parser.in_parens():
        value = parser.parse_integer()

    assert value == 1
    assert parser.pos == 3


def test_wrong_delimiter():
    parser = Parser(Context(), "(1)")
    with pytest.raises(ParseError, match="'<' expected"):
        with parser.in_angle_brackets():
            parser.parse_integer()


def test_early_return_delimiter():
    def my_parse(parser: Parser):
        with parser.in_angle_brackets():
            if parser.parse_integer() == 1:
                return 1
            return 2

    parser = Parser(Context(), "<1>")
    assert my_parse(parser) == 1
    assert parser.pos == 3


class MySingletonEnum(StrEnum):
    A = "a"


def test_parse_singleton_enum_fail():
    parser = Parser(Context(), "b")
    with pytest.raises(ParseError, match="Expected `a`"):
        parser.parse_str_enum(MySingletonEnum)


def test_metadata_parsing():
    ctx = Context()
    ctx.register_dialect("test", lambda: Test)
    metadata_dict = '{-# dialect_resources: {test: {some_res: "0x1"}} #-}'

    parser = Parser(ctx, metadata_dict)
    assert parser._parse_file_metadata_dictionary() is None

    test_dialect = ctx.get_dialect("test")
    interface = test_dialect.get_interface(OpAsmDialectInterface)
    assert interface

    element = interface.lookup("some_res")
    assert element == "0x1"


@pytest.mark.parametrize(
    "input, expected",
    [
        ("a", "a"),
        ('"a"', "a"),
        ("a-b", "a"),
        ('"a-b"', "a-b"),
        ("2a", None),
    ],
)
def test_parse_identifier_or_str_literal(input: str, expected: str | None):
    parser = Parser(Context(), input)
    result = parser.parse_optional_identifier_or_str_literal()
    assert result == expected

    parser = Parser(Context(), input)
    if expected is None:
        with pytest.raises(ParseError, match="identifier or string literal expected"):
            parser.parse_identifier_or_str_literal()
    else:
        assert parser.parse_identifier_or_str_literal() == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("", []),
        ("2x3x4", [2, 3, 4]),
        ("9x1x5x", [9, 1, 5]),
        ("9x?x1x?", [9, DYNAMIC_INDEX, 1, DYNAMIC_INDEX]),
    ],
)
def test_parse_dimension_list(input: str, expected: list[int]):
    parser = Parser(Context(), input)

    result = parser.parse_dimension_list()
    assert result == expected
