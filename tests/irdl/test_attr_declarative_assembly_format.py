"""Tests for the attribute/type declarative assembly format.

These tests drive ``AttrFormatProgram`` directly, covering the empty format and
the structural directives (whitespace, punctuation, keyword).
"""

from __future__ import annotations

from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.irdl.declarative_assembly_format import (
    AttrFormatProgram,
    AttrWhitespaceDirective,
)
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


def test_empty_format_produces_empty_program():
    assert _program("").stmts == ()


def test_empty_format_prints_and_parses_nothing():
    assert _print("") == ""
    assert _parse("", "") == []


def test_parse_attribute_end_to_end():
    ctx = Context()
    ctx.load_attr_or_type(EmptyType)
    assert Parser(ctx, "!test_af.empty").parse_type() == EmptyType()


def test_error_unexpected_token():
    with pytest.raises(ParseError, match="unexpected token"):
        _program("$foo")


# Whitespace directive


@pytest.mark.parametrize(
    "format_str, expected",
    [("` `", " "), ("``", ""), ("`\\n`", "\n")],
)
def test_whitespace_print(format_str: str, expected: str):
    assert _print(format_str) == expected


def test_whitespace_parses_to_directive_and_consumes_nothing():
    assert _program("` `").stmts == (AttrWhitespaceDirective(" "),)
    assert _parse("` `", "") == []


def test_error_invalid_whitespace():
    with pytest.raises(ParseError, match="unexpected whitespace in directive"):
        _program("`  `")


def test_error_not_a_whitespace_directive():
    with pytest.raises(ParseError, match="expected a whitespace directive"):
        _program("`+`")
