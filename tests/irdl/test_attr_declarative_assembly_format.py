"""Tests for declarative assembly format for ParametrizedAttribute types."""

from __future__ import annotations

from io import StringIO

import pytest

from xdsl.dialects.builtin import IntegerType, i32, i64
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import ParamAttrDef, irdl_attr_definition, param_def
from xdsl.irdl.declarative_assembly_format import AttrFormatProgram
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError

# ============================================================================
# Test attribute definitions (used for constructing ParamAttrDef manually)
# ============================================================================


@irdl_attr_definition
class SimpleType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.simple"
    value: IntegerType = param_def()


@irdl_attr_definition
class TwoParamType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.two_param"
    first: IntegerType = param_def()
    second: IntegerType = param_def()


@irdl_attr_definition
class KeywordType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.keyword"
    value: IntegerType = param_def()


# ============================================================================
# Helpers
# ============================================================================


def _attr_def_for(cls: type[ParametrizedAttribute]) -> ParamAttrDef:
    """Build a ParamAttrDef from a ParametrizedAttribute class."""
    return ParamAttrDef.from_pyrdl(cls)


def _print_with_format(
    format_str: str, attr: ParametrizedAttribute, attr_def: ParamAttrDef
) -> str:
    program = AttrFormatProgram.from_str(format_str, attr_def)
    output = StringIO()
    printer = Printer(stream=output)
    program.print(printer, attr)
    return output.getvalue()


def _parse_with_format(
    format_str: str,
    body: str,
    attr_def: ParamAttrDef,
) -> list[Attribute]:
    from xdsl.context import Context

    ctx = Context(allow_unregistered=True)
    parser = Parser(ctx, body)
    program = AttrFormatProgram.from_str(format_str, attr_def)
    return program.parse(parser, attr_def)


# ============================================================================
# Print tests
# ============================================================================


def test_print_single_param():
    attr_def = _attr_def_for(SimpleType)
    result = _print_with_format("$value", SimpleType(i32), attr_def)
    assert result == "i32"


def test_print_two_params():
    attr_def = _attr_def_for(TwoParamType)
    result = _print_with_format("$first `,` $second", TwoParamType(i32, i64), attr_def)
    assert result == "i32, i64"


def test_print_keyword():
    attr_def = _attr_def_for(KeywordType)
    result = _print_with_format("`stride` `=` $value", KeywordType(i32), attr_def)
    assert result == "stride = i32"


def test_print_whitespace_suppress():
    attr_def = _attr_def_for(TwoParamType)
    result = _print_with_format("$first `` $second", TwoParamType(i32, i64), attr_def)
    assert result == "i32i64"


# ============================================================================
# Parse tests
# ============================================================================


def test_parse_single_param():
    attr_def = _attr_def_for(SimpleType)
    params = _parse_with_format("$value", "i32", attr_def)
    assert params == [i32]


def test_parse_two_params():
    attr_def = _attr_def_for(TwoParamType)
    params = _parse_with_format("$first `,` $second", "i32, i64", attr_def)
    assert params == [i32, i64]


def test_parse_keyword():
    attr_def = _attr_def_for(KeywordType)
    params = _parse_with_format("`stride` `=` $value", "stride = i32", attr_def)
    assert params == [i32]


# ============================================================================
# Round-trip tests (print then parse)
# ============================================================================


@pytest.mark.parametrize(
    "fmt, attr, attr_cls",
    [
        ("$value", SimpleType(i32), SimpleType),
        ("$value", SimpleType(i64), SimpleType),
        ("$first `,` $second", TwoParamType(i32, i64), TwoParamType),
        ("`stride` `=` $value", KeywordType(i32), KeywordType),
    ],
)
def test_roundtrip(
    fmt: str, attr: ParametrizedAttribute, attr_cls: type[ParametrizedAttribute]
):
    attr_def = _attr_def_for(attr_cls)
    printed = _print_with_format(fmt, attr, attr_def)
    parsed = _parse_with_format(fmt, printed, attr_def)
    reconstructed = attr_cls(*parsed)
    assert reconstructed == attr


# ============================================================================
# Error tests (format string validation)
# ============================================================================


def test_error_missing_parameter():
    attr_def = _attr_def_for(TwoParamType)
    with pytest.raises(ParseError, match="parameter 'second' not found"):
        AttrFormatProgram.from_str("$first", attr_def)


def test_error_duplicate_parameter():
    attr_def = _attr_def_for(SimpleType)
    with pytest.raises(ParseError, match="is already bound"):
        AttrFormatProgram.from_str("$value `,` $value", attr_def)


def test_error_unknown_variable():
    attr_def = _attr_def_for(SimpleType)
    with pytest.raises(ParseError, match="does not refer to a parameter"):
        AttrFormatProgram.from_str("$nonexistent", attr_def)
