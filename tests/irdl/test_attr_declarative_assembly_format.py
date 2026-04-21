"""Tests for declarative assembly format for ParametrizedAttribute types."""

from __future__ import annotations

from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    IntegerType,
    NoneAttr,
    i1,
    i32,
    i64,
)
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import ParamAttrDef, irdl_attr_definition, param_def
from xdsl.irdl.declarative_assembly_format import AttrFormatProgram
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, PyRDLAttrDefinitionError

# ============================================================================
# Test attribute definitions (without assembly_format — for unit tests)
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
# Test attribute definitions (with assembly_format — for integration tests)
# ============================================================================


@irdl_attr_definition
class SimpleIntegType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.simple_i"
    value: IntegerType = param_def()
    assembly_format = "$value"


@irdl_attr_definition
class TwoParamIntegType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.two_param_i"
    first: IntegerType = param_def()
    second: IntegerType = param_def()
    assembly_format = "$first `,` $second"


@irdl_attr_definition
class KeywordIntegType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.keyword_i"
    value: IntegerType = param_def()
    assembly_format = "`stride` `=` $value"


@irdl_attr_definition
class InnerType(ParametrizedAttribute, TypeAttribute):
    """An inner type with its own assembly_format, for composition testing."""

    name = "test_af.inner"
    elem: IntegerType = param_def()
    assembly_format = "$elem"


@irdl_attr_definition
class OuterType(ParametrizedAttribute, TypeAttribute):
    """Contains an InnerType parameter to test sub-attribute composition."""

    name = "test_af.outer"
    wrapped: InnerType = param_def()
    tag: IntegerType = param_def()
    assembly_format = "$wrapped `,` $tag"


@irdl_attr_definition
class WrapperType(ParametrizedAttribute, TypeAttribute):
    """Wraps a generic Attribute — accepts any type including complex ones."""

    name = "test_af.wrapper"
    inner: Attribute = param_def()
    assembly_format = "$inner"


@irdl_attr_definition
class PairType(ParametrizedAttribute, TypeAttribute):
    """A pair of generic attributes."""

    name = "test_af.pair"
    left: Attribute = param_def()
    right: Attribute = param_def()
    assembly_format = "$left `,` $right"


@irdl_attr_definition
class PairAttr(ParametrizedAttribute):
    """A pair attribute (not a type) — uses # prefix."""

    name = "test_af.pair_attr"
    left: Attribute = param_def()
    right: Attribute = param_def()
    assembly_format = "$left `,` $right"


@irdl_attr_definition
class ThreeParamType(ParametrizedAttribute, TypeAttribute):
    """Three params for more complex tests."""

    name = "test_af.three_param"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    c: IntegerType = param_def()
    assembly_format = "$a `,` $b `,` $c"


@irdl_attr_definition
class OptionalParamType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.optional_param"
    value: IntegerType = param_def()
    opt: IntegerType | NoneAttr = param_def()
    assembly_format = "$value (`,` $opt^)?"


@irdl_attr_definition
class ElseBranchType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.else_branch"
    a: IntegerType | NoneAttr = param_def()
    b: IntegerType | NoneAttr = param_def()
    assembly_format = "($a^) : (`fallback` $b)?"


# ============================================================================
# Unit test helpers (call AttrFormatProgram directly, no @irdl_attr_definition)
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
    ctx = Context(allow_unregistered=True)
    parser = Parser(ctx, body)
    program = AttrFormatProgram.from_str(format_str, attr_def)
    return program.parse(parser, attr_def)


# ============================================================================
# Integration test helpers (parse/print via operation parser)
# ============================================================================


def parse_type(type_str: str, ctx: Context) -> Attribute:
    """Parse a type like '!test_af.simple_i<i32>' and return the type attribute."""
    program = f'"test.op"() : () -> {type_str}'
    parser = Parser(ctx, program)
    op = parser.parse_optional_operation()
    assert op is not None, f"Failed to parse operation with type {type_str}"
    return op.results[0].type


def parse_attr(attr_str: str, ctx: Context) -> Attribute:
    """Parse an attribute like '#test_af.pair_attr<i32, i64>'."""
    program = f'"test.op"() {{attr = {attr_str}}} : () -> ()'
    parser = Parser(ctx, program)
    op = parser.parse_optional_operation()
    assert op is not None, f"Failed to parse operation with attr {attr_str}"
    return op.attributes["attr"]


def print_attr(attr: Attribute) -> str:
    """Print an attribute to a string."""
    output = StringIO()
    printer = Printer(stream=output)
    printer.print_attribute(attr)
    return output.getvalue()


def check_roundtrip(
    attr_type: type[ParametrizedAttribute], body: str, ctx: Context
) -> None:
    """Parse !type<body>, print it, verify the body matches."""
    type_str = f"!{attr_type.name}<{body}>"
    parsed = parse_type(type_str, ctx)
    printed = print_attr(parsed)
    assert printed == type_str, f"Round-trip failed: {printed!r} != {type_str!r}"


def check_attr_roundtrip(
    attr_type: type[ParametrizedAttribute], body: str, ctx: Context
) -> None:
    """Round-trip for #-prefixed attributes."""
    attr_str = f"#{attr_type.name}<{body}>"
    parsed = parse_attr(attr_str, ctx)
    printed = print_attr(parsed)
    assert printed == attr_str, f"Attr round-trip failed: {printed!r} != {attr_str!r}"


# ============================================================================
# Fixture
# ============================================================================


@pytest.fixture
def ctx() -> Context:
    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(SimpleIntegType)
    ctx.load_attr_or_type(TwoParamIntegType)
    ctx.load_attr_or_type(KeywordIntegType)
    ctx.load_attr_or_type(InnerType)
    ctx.load_attr_or_type(OuterType)
    ctx.load_attr_or_type(WrapperType)
    ctx.load_attr_or_type(PairType)
    ctx.load_attr_or_type(PairAttr)
    ctx.load_attr_or_type(ThreeParamType)
    ctx.load_attr_or_type(OptionalParamType)
    ctx.load_attr_or_type(ElseBranchType)
    return ctx


# ============================================================================
# Unit tests — AttrFormatProgram directly (print)
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
# Unit tests — AttrFormatProgram directly (parse)
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
# Unit tests — round-trip (print then parse)
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
# Unit tests — format string validation errors
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


# ============================================================================
# Integration tests — basic round-trips via operation parser
# ============================================================================


@pytest.mark.parametrize("body", ["i32", "i64"])
def test_simple_param_roundtrip(body: str, ctx: Context):
    check_roundtrip(SimpleIntegType, body, ctx)


@pytest.mark.parametrize("body", ["i32, i64", "i1, i32"])
def test_two_param_roundtrip(body: str, ctx: Context):
    check_roundtrip(TwoParamIntegType, body, ctx)


@pytest.mark.parametrize("body", ["stride = i32", "stride = i64"])
def test_keyword_roundtrip(body: str, ctx: Context):
    check_roundtrip(KeywordIntegType, body, ctx)


def test_three_param_roundtrip(ctx: Context):
    check_roundtrip(ThreeParamType, "i1, i32, i64", ctx)


# ============================================================================
# Integration tests — sub-attribute composition
# ============================================================================


@pytest.mark.parametrize(
    "body",
    [
        "!test_af.inner<i32>, i64",
        "!test_af.inner<i1>, i32",
    ],
)
def test_outer_type_roundtrip(body: str, ctx: Context):
    check_roundtrip(OuterType, body, ctx)


def test_outer_type_constructs_correctly(ctx: Context):
    """Verify the outer type correctly nests the inner type."""
    parsed = parse_type("!test_af.outer<!test_af.inner<i32>, i64>", ctx)
    assert isinstance(parsed, OuterType)
    assert isinstance(parsed.wrapped, InnerType)
    assert parsed.wrapped.elem == i32
    assert parsed.tag == i64


# ============================================================================
# Integration tests — complex nesting
# ============================================================================


@pytest.mark.parametrize(
    "body",
    [
        "i32",
        "f32",
        "vector<4xi32>",
        "tensor<2x3xf32>",
    ],
)
def test_wrapper_complex_types(body: str, ctx: Context):
    check_roundtrip(WrapperType, body, ctx)


@pytest.mark.parametrize(
    "body",
    [
        "i32, i64",
        "tensor<2x3xf32>, memref<16xi32>",
    ],
)
def test_pair_complex_types(body: str, ctx: Context):
    check_roundtrip(PairType, body, ctx)


def test_deeply_nested_types(ctx: Context):
    """Wrapper wrapping a pair wrapping inner types."""
    check_roundtrip(
        WrapperType,
        "!test_af.pair<!test_af.inner<i32>, !test_af.inner<i64>>",
        ctx,
    )


# ============================================================================
# Integration tests — attribute parameters (# prefix)
# ============================================================================


def test_pair_attr_roundtrip(ctx: Context):
    check_attr_roundtrip(PairAttr, "i32, i64", ctx)


def test_pair_attr_constructs_correctly(ctx: Context):
    parsed = parse_attr("#test_af.pair_attr<i32, i64>", ctx)
    assert isinstance(parsed, PairAttr)
    assert parsed.left == i32
    assert parsed.right == i64


def test_wrapper_with_attr_parameter(ctx: Context):
    """A type parameter that is a #-prefixed attribute."""
    check_roundtrip(WrapperType, "#test_af.pair_attr<i32, i64>", ctx)


# ============================================================================
# Integration tests — whitespace directives
# ============================================================================


def test_whitespace_suppress():
    """`` `` (empty backtick) suppresses whitespace."""

    @irdl_attr_definition
    class SuppressType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.suppress"
        a: IntegerType = param_def()
        b: IntegerType = param_def()
        assembly_format = "$a `` $b"

    attr = SuppressType(i32, i64)
    printed = print_attr(attr)
    assert printed == "!test_af.suppress<i32i64>"


def test_whitespace_newline():
    r"""`\n` inserts newline."""

    @irdl_attr_definition
    class NewlineType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.newline"
        a: IntegerType = param_def()
        b: IntegerType = param_def()
        assembly_format = "$a `` `\\n` `` $b"

    attr = NewlineType(i32, i64)
    printed = print_attr(attr)
    assert printed == "!test_af.newline<i32\ni64>"


# ============================================================================
# Integration tests — optional groups
# ============================================================================


@pytest.mark.parametrize("body", ["i32, i64", "i32"])
def test_optional_param_roundtrip(body: str, ctx: Context):
    check_roundtrip(OptionalParamType, body, ctx)


def test_optional_param_present_constructs(ctx: Context):
    parsed = parse_type("!test_af.optional_param<i32, i64>", ctx)
    assert isinstance(parsed, OptionalParamType)
    assert parsed.value == i32
    assert parsed.opt == i64


def test_optional_param_absent_constructs(ctx: Context):
    parsed = parse_type("!test_af.optional_param<i32>", ctx)
    assert isinstance(parsed, OptionalParamType)
    assert parsed.value == i32
    assert isinstance(parsed.opt, NoneAttr)


def test_else_branch_then(ctx: Context):
    check_roundtrip(ElseBranchType, "i32", ctx)


def test_else_branch_else(ctx: Context):
    check_roundtrip(ElseBranchType, "fallback i64", ctx)


def test_else_branch_then_constructs(ctx: Context):
    parsed = parse_type("!test_af.else_branch<i32>", ctx)
    assert isinstance(parsed, ElseBranchType)
    assert parsed.a == i32
    assert isinstance(parsed.b, NoneAttr)


@pytest.mark.parametrize(
    "attr, expected",
    [
        (OptionalParamType(i32, i64), "!test_af.optional_param<i32, i64>"),
        (OptionalParamType(i32, NoneAttr()), "!test_af.optional_param<i32>"),
        (ElseBranchType(i32, NoneAttr()), "!test_af.else_branch<i32>"),
        (ElseBranchType(NoneAttr(), i64), "!test_af.else_branch<fallback i64>"),
    ],
)
def test_optional_print_correctness(attr: Attribute, expected: str):
    assert print_attr(attr) == expected


# ============================================================================
# Integration tests — qualified directive
# ============================================================================


def test_qualified_roundtrip():
    """qualified($x) should parse and print identically to $x in xdsl."""

    @irdl_attr_definition
    class QualifiedType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.qualified"
        value: IntegerType = param_def()
        assembly_format = "qualified($value)"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(QualifiedType)
    check_roundtrip(QualifiedType, "i32", ctx)


# ============================================================================
# Integration tests — params directive
# ============================================================================


def test_params_directive_roundtrip():
    """params captures all parameters, printed comma-separated."""

    @irdl_attr_definition
    class ParamsType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.params_all"
        a: IntegerType = param_def()
        b: IntegerType = param_def()
        assembly_format = "params"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(ParamsType)
    check_roundtrip(ParamsType, "i32, i64", ctx)


def test_params_directive_three_params():
    """params with three parameters."""

    @irdl_attr_definition
    class Params3Type(ParametrizedAttribute, TypeAttribute):
        name = "test_af.params3"
        a: IntegerType = param_def()
        b: IntegerType = param_def()
        c: IntegerType = param_def()
        assembly_format = "params"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(Params3Type)
    check_roundtrip(Params3Type, "i1, i32, i64", ctx)


def test_params_with_optional():
    """params with optional parameter omitted."""

    @irdl_attr_definition
    class ParamsOptType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.params_opt"
        req: IntegerType = param_def()
        opt: IntegerType | NoneAttr = param_def()
        assembly_format = "params"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(ParamsOptType)
    check_roundtrip(ParamsOptType, "i32, i64", ctx)
    check_roundtrip(ParamsOptType, "i32", ctx)


# ============================================================================
# Integration tests — struct directive
# ============================================================================


def test_struct_all_params():
    """struct(params) prints all params as key=value pairs."""

    @irdl_attr_definition
    class StructAllType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.struct_all"
        x: IntegerType = param_def()
        y: IntegerType = param_def()
        assembly_format = "struct(params)"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(StructAllType)
    check_roundtrip(StructAllType, "x = i32, y = i64", ctx)


def test_struct_subset():
    """struct($a, $b) with a subset of parameters."""

    @irdl_attr_definition
    class StructSubType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.struct_sub"
        a: IntegerType = param_def()
        b: IntegerType = param_def()
        c: IntegerType = param_def()
        assembly_format = "struct($a, $b) `,` $c"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(StructSubType)
    check_roundtrip(StructSubType, "a = i32, b = i64, i1", ctx)


def test_struct_reordered_parse():
    """Struct fields can be parsed in any order."""

    @irdl_attr_definition
    class StructReorderType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.struct_reorder"
        a: IntegerType = param_def()
        b: IntegerType = param_def()
        assembly_format = "struct(params)"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(StructReorderType)
    parsed = parse_type("!test_af.struct_reorder<b = i64, a = i32>", ctx)
    assert isinstance(parsed, StructReorderType)
    assert parsed.a == i32
    assert parsed.b == i64


def test_struct_optional_param():
    """Struct with optional parameter omitted."""

    @irdl_attr_definition
    class StructOptType(ParametrizedAttribute, TypeAttribute):
        name = "test_af.struct_opt"
        req: IntegerType = param_def()
        opt: IntegerType | NoneAttr = param_def()
        assembly_format = "struct(params)"

    ctx = Context(allow_unregistered=True)
    ctx.load_attr_or_type(StructOptType)
    check_roundtrip(StructOptType, "req = i32, opt = i64", ctx)
    check_roundtrip(StructOptType, "req = i32", ctx)


# ============================================================================
# Integration tests — printing correctness
# ============================================================================


@pytest.mark.parametrize(
    "attr, expected",
    [
        (SimpleIntegType(i32), "!test_af.simple_i<i32>"),
        (TwoParamIntegType(i32, i64), "!test_af.two_param_i<i32, i64>"),
        (KeywordIntegType(i32), "!test_af.keyword_i<stride = i32>"),
        (ThreeParamType(i1, i32, i64), "!test_af.three_param<i1, i32, i64>"),
        (PairAttr(i32, i64), "#test_af.pair_attr<i32, i64>"),
    ],
)
def test_print_correctness(attr: Attribute, expected: str):
    assert print_attr(attr) == expected


# ============================================================================
# Integration tests — definition-time errors
# ============================================================================


def test_defn_error_missing_parameter():
    """Format string missing a parameter should fail at definition time."""
    with pytest.raises(PyRDLAttrDefinitionError, match="parameter 'second' not found"):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_missing"
            first: IntegerType = param_def()
            second: IntegerType = param_def()
            assembly_format = "$first"


def test_defn_error_duplicate_parameter():
    """Duplicate parameter in format string should fail at definition time."""
    with pytest.raises(PyRDLAttrDefinitionError, match="is already bound"):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_dup"
            value: IntegerType = param_def()
            assembly_format = "$value `,` $value"


def test_defn_error_unknown_variable():
    """Unknown variable in format string should fail at definition time."""
    with pytest.raises(PyRDLAttrDefinitionError, match="does not refer to a parameter"):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_unknown"
            value: IntegerType = param_def()
            assembly_format = "$nonexistent"


def test_defn_error_assembly_format_with_parse_parameters():
    """assembly_format with custom parse_parameters should fail."""
    with pytest.raises(
        PyRDLAttrDefinitionError,
        match="Cannot use assembly_format with custom parse_parameters",
    ):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_custom_parse"
            value: IntegerType = param_def()
            assembly_format = "$value"

            @classmethod
            def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
                return []


def test_defn_error_assembly_format_with_print_parameters():
    """assembly_format with custom print_parameters should fail."""
    with pytest.raises(
        PyRDLAttrDefinitionError,
        match="Cannot use assembly_format with custom print_parameters",
    ):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_custom_print"
            value: IntegerType = param_def()
            assembly_format = "$value"

            def print_parameters(self, printer: Printer) -> None:
                pass


def test_defn_error_assembly_format_on_singleton():
    """assembly_format on attribute with no parameters should fail."""
    with pytest.raises(
        PyRDLAttrDefinitionError,
        match="Cannot use assembly_format on attribute with no parameters",
    ):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_singleton"
            assembly_format = ""
