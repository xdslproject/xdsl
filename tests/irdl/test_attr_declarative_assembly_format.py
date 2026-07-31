"""
Tests for the attribute/type declarative assembly format.

These tests drive ``AttrFormatProgram`` directly, covering the empty format and
the structural directives (whitespace, punctuation, keyword).
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


def test_empty_format_produces_empty_program():
    assert _program("").stmts == ()


@pytest.mark.parametrize(
    "format_str, body, printed",
    [
        ("", "", ""),
        ("` `", "", " "),
        ("``", "", ""),
        ("`\\n`", "", "\n"),
        ("`hello`", "hello", "hello"),
        ("`foo``bar`", "foo bar", "foo bar"),
    ],
)
def test_print_and_parse(format_str: str, body: str, printed: str):
    assert _print(format_str) == printed
    assert _parse(format_str, body) == []


def test_parse_attribute_end_to_end():
    ctx = Context()
    ctx.load_attr_or_type(EmptyType)
    assert Parser(ctx, "!test_af.empty").parse_type() == EmptyType()


@pytest.mark.parametrize(
    "format_str, error",
    [
        ("$foo", "unexpected token"),
        ("`  `", "unexpected whitespace in directive"),
        ("`+`", "punctuation or identifier expected"),
    ],
)
def test_error(format_str: str, error: str):
    with pytest.raises(ParseError, match=error):
        _program(format_str)
