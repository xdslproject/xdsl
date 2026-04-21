"""Tests for the structural directives of the attribute/type assembly format.

These tests exercise the whitespace, punctuation and keyword directives in
isolation (without parameters), driving ``AttrFormatProgram`` directly.
"""

from __future__ import annotations

from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.irdl.declarative_assembly_format import AttrFormatProgram
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError


@irdl_attr_definition
class EmptyType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.empty"


def _program(format_str: str) -> AttrFormatProgram:
    return AttrFormatProgram.from_str(format_str, EmptyType.get_irdl_definition())


def _print(format_str: str) -> str:
    output = StringIO()
    _program(format_str).print(Printer(stream=output), EmptyType())
    return output.getvalue()


def _parse(format_str: str, body: str) -> list[Attribute]:
    parser = Parser(Context(allow_unregistered=True), body)
    return _program(format_str).parse(parser, EmptyType.get_irdl_definition())


# ============================================================================
# Whitespace directive
# ============================================================================


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("` `", " "),
        ("``", ""),
        ("`\\n`", "\n"),
    ],
)
def test_print_whitespace(fmt: str, expected: str):
    assert _print(fmt) == expected


def test_parse_whitespace_is_noop():
    assert _parse("` `", "") == []


def test_error_invalid_whitespace():
    with pytest.raises(ParseError, match="unexpected whitespace in directive"):
        _program("`  `")


# ============================================================================
# Keyword directive
# ============================================================================


def test_print_keyword():
    assert _print("`hello`") == "hello"


def test_print_two_keywords_spaced():
    assert _print("`hello` `world`") == "hello world"


def test_parse_keyword():
    assert _parse("`hello`", "hello") == []


# ============================================================================
# Punctuation directive
# ============================================================================


def test_print_single_punctuation():
    assert _print("`<`") == "<"


def test_print_punctuation_after_punctuation_spaced():
    # last_was_punctuation is True, `+` is not a closer -> space inserted
    assert _print("`+` `+`") == "+ +"


def test_print_punctuation_after_punctuation_closer():
    # last_was_punctuation is True, `>` is a closer -> no space
    assert _print("`+` `>`") == "+>"


def test_print_punctuation_after_keyword_spaced():
    # last_was_punctuation is False, `+` is not a bracket/comma -> space
    assert _print("`kw` `+`") == "kw +"


def test_print_punctuation_after_keyword_comma():
    # last_was_punctuation is False, `,` needs no leading space
    assert _print("`kw` `,`") == "kw,"


def test_parse_punctuation():
    assert _parse("`+`", "+") == []


# ============================================================================
# Format-string errors
# ============================================================================


def test_error_unexpected_token():
    with pytest.raises(ParseError, match="unexpected token"):
        _program("$foo")


def test_error_not_punctuation_or_identifier():
    with pytest.raises(ParseError, match="punctuation or identifier expected"):
        _program("`1`")
