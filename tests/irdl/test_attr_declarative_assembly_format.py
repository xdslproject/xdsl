"""Tests for the attribute/type declarative assembly format plumbing.

These tests drive ``AttrFormatProgram`` directly. At this stage only the empty
format is supported; concrete directives are added in later commits.
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


def test_empty_format_is_valid():
    program = _program("")
    assert program.stmts == ()
    assert _print("") == ""
    assert _parse("", "") == []


def test_error_unexpected_token():
    with pytest.raises(ParseError, match="unexpected token"):
        _program("$foo")
