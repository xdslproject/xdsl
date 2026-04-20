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
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.irdl.declarative_assembly_format import (
    AttrCustomDirective,
    AttrParsingState,
    ParameterVariable,
    PrintingState,
    irdl_attr_custom_directive,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, PyRDLAttrDefinitionError

# ============================================================================
# Test attribute definitions
# ============================================================================


@irdl_attr_definition
class SimpleType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.simple"
    value: IntegerType = param_def()
    assembly_format = "$value"


@irdl_attr_definition
class TwoParamType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.two_param"
    first: IntegerType = param_def()
    second: IntegerType = param_def()
    assembly_format = "$first `,` $second"


@irdl_attr_definition
class OptionalParamType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.optional"
    elem: IntegerType = param_def()
    layout: NoneAttr | IntegerType = param_def()
    assembly_format = "$elem (`,` $layout^)?"


@irdl_attr_definition
class KeywordType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.keyword"
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
class DoubleOptionalType(ParametrizedAttribute, TypeAttribute):
    """Type with two independent optional groups."""

    name = "test_af.double_optional"
    base: IntegerType = param_def()
    opt_a: NoneAttr | IntegerType = param_def()
    opt_b: NoneAttr | IntegerType = param_def()
    assembly_format = "$base (`a` `=` $opt_a^)? (`b` `=` $opt_b^)?"


@irdl_attr_definition
class ElseBranchType(ParametrizedAttribute, TypeAttribute):
    """Type with an optional group that has an else branch."""

    name = "test_af.else_branch"
    base: IntegerType = param_def()
    opt: NoneAttr | IntegerType = param_def()
    fallback: NoneAttr | IntegerType = param_def()
    assembly_format = "$base (`opt` `=` $opt^) : (`fallback` `=` $fallback)?"


@irdl_attr_definition
class ThreeParamType(ParametrizedAttribute, TypeAttribute):
    """Three params for more complex tests."""

    name = "test_af.three_param"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    c: IntegerType = param_def()
    assembly_format = "$a `,` $b `,` $c"


# ============================================================================
# Helpers
# ============================================================================


def parse_type(type_str: str, ctx: Context) -> Attribute:
    """Parse a type like '!test_af.simple<i32>' and return the type attribute."""
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
    ctx.load_attr_or_type(SimpleType)
    ctx.load_attr_or_type(TwoParamType)
    ctx.load_attr_or_type(OptionalParamType)
    ctx.load_attr_or_type(KeywordType)
    ctx.load_attr_or_type(InnerType)
    ctx.load_attr_or_type(OuterType)
    ctx.load_attr_or_type(WrapperType)
    ctx.load_attr_or_type(PairType)
    ctx.load_attr_or_type(PairAttr)
    ctx.load_attr_or_type(DoubleOptionalType)
    ctx.load_attr_or_type(ElseBranchType)
    ctx.load_attr_or_type(ThreeParamType)
    return ctx


# ============================================================================
# Basic round-trip tests
# ============================================================================


@pytest.mark.parametrize("body", ["i32", "i64"])
def test_simple_param_roundtrip(body: str, ctx: Context):
    check_roundtrip(SimpleType, body, ctx)


@pytest.mark.parametrize("body", ["i32, i64", "i1, i32"])
def test_two_param_roundtrip(body: str, ctx: Context):
    check_roundtrip(TwoParamType, body, ctx)


def test_optional_param_absent(ctx: Context):
    check_roundtrip(OptionalParamType, "i32", ctx)


def test_optional_param_present(ctx: Context):
    check_roundtrip(OptionalParamType, "i32, i64", ctx)


def test_optional_constructs_correctly(ctx: Context):
    """Absent optional parameter becomes NoneAttr; present one is parsed."""
    absent = parse_type("!test_af.optional<i32>", ctx)
    assert isinstance(absent, OptionalParamType)
    assert absent.elem == i32
    assert isinstance(absent.layout, NoneAttr)

    present = parse_type("!test_af.optional<i32, i64>", ctx)
    assert isinstance(present, OptionalParamType)
    assert present.elem == i32
    assert present.layout == i64


@pytest.mark.parametrize("body", ["stride = i32", "stride = i64"])
def test_keyword_roundtrip(body: str, ctx: Context):
    check_roundtrip(KeywordType, body, ctx)


def test_three_param_roundtrip(ctx: Context):
    check_roundtrip(ThreeParamType, "i1, i32, i64", ctx)


# ============================================================================
# Sub-attribute composition (inner type with its own assembly_format)
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
# Complex nesting (builtin complex types as parameters)
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
# Attribute parameters (# prefix, not just ! types)
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
# Multiple optional groups
# ============================================================================


@pytest.mark.parametrize(
    "body, expected_a, expected_b",
    [
        ("i32", NoneAttr, NoneAttr),
        ("i32 a = i64", IntegerType, NoneAttr),
        ("i32 b = i64", NoneAttr, IntegerType),
        ("i32 a = i64 b = i1", IntegerType, IntegerType),
    ],
)
def test_double_optional_group(
    body: str,
    expected_a: type,
    expected_b: type,
    ctx: Context,
):
    parsed = parse_type(f"!test_af.double_optional<{body}>", ctx)
    assert isinstance(parsed, DoubleOptionalType)
    assert isinstance(parsed.opt_a, expected_a)
    assert isinstance(parsed.opt_b, expected_b)


@pytest.mark.parametrize(
    "body",
    [
        "i32",
        "i32 a = i64",
        "i32 b = i64",
        "i32 a = i64 b = i1",
    ],
)
def test_double_optional_roundtrip(body: str, ctx: Context):
    check_roundtrip(DoubleOptionalType, body, ctx)


# ============================================================================
# Else branch in optional groups
# ============================================================================


def test_else_branch_then_path(ctx: Context):
    """When optional is present, then-branch is taken."""
    parsed = parse_type("!test_af.else_branch<i32 opt = i64>", ctx)
    assert isinstance(parsed, ElseBranchType)
    assert parsed.opt == i64
    assert isinstance(parsed.fallback, NoneAttr)


def test_else_branch_else_path(ctx: Context):
    """When optional is absent, else-branch is taken."""
    parsed = parse_type("!test_af.else_branch<i32 fallback = i64>", ctx)
    assert isinstance(parsed, ElseBranchType)
    assert isinstance(parsed.opt, NoneAttr)
    assert parsed.fallback == i64


@pytest.mark.parametrize("body", ["i32 opt = i64", "i32 fallback = i64"])
def test_else_branch_roundtrip(body: str, ctx: Context):
    check_roundtrip(ElseBranchType, body, ctx)


# ============================================================================
# Whitespace directives
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
# Printing correctness — construct programmatically and verify output
# ============================================================================


@pytest.mark.parametrize(
    "attr, expected",
    [
        (SimpleType(i32), "!test_af.simple<i32>"),
        (TwoParamType(i32, i64), "!test_af.two_param<i32, i64>"),
        (OptionalParamType(i32, NoneAttr()), "!test_af.optional<i32>"),
        (OptionalParamType(i32, i64), "!test_af.optional<i32, i64>"),
        (KeywordType(i32), "!test_af.keyword<stride = i32>"),
        (ThreeParamType(i1, i32, i64), "!test_af.three_param<i1, i32, i64>"),
        (PairAttr(i32, i64), "#test_af.pair_attr<i32, i64>"),
    ],
)
def test_print_correctness(attr: Attribute, expected: str):
    assert print_attr(attr) == expected


@pytest.mark.parametrize(
    "attr, expected",
    [
        (
            DoubleOptionalType(i32, NoneAttr(), NoneAttr()),
            "!test_af.double_optional<i32>",
        ),
        (
            DoubleOptionalType(i32, i64, NoneAttr()),
            "!test_af.double_optional<i32 a = i64>",
        ),
        (
            DoubleOptionalType(i32, NoneAttr(), i64),
            "!test_af.double_optional<i32 b = i64>",
        ),
        (
            DoubleOptionalType(i32, i64, i1),
            "!test_af.double_optional<i32 a = i64 b = i1>",
        ),
    ],
)
def test_print_double_optional(attr: Attribute, expected: str):
    assert print_attr(attr) == expected


@pytest.mark.parametrize(
    "attr, expected",
    [
        (
            ElseBranchType(i32, i64, NoneAttr()),
            "!test_af.else_branch<i32 opt = i64>",
        ),
        (
            ElseBranchType(i32, NoneAttr(), i64),
            "!test_af.else_branch<i32 fallback = i64>",
        ),
    ],
)
def test_print_else_branch(attr: Attribute, expected: str):
    assert print_attr(attr) == expected


# ============================================================================
# Error tests
# ============================================================================


def test_error_missing_parameter():
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


def test_error_duplicate_parameter():
    """Duplicate parameter in format string should fail at definition time."""
    with pytest.raises(PyRDLAttrDefinitionError, match="is already bound"):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_dup"
            value: IntegerType = param_def()
            assembly_format = "$value `,` $value"


def test_error_unknown_variable():
    """Unknown variable in format string should fail at definition time."""
    with pytest.raises(PyRDLAttrDefinitionError, match="does not refer to a parameter"):

        @irdl_attr_definition
        class BadType(  # pyright: ignore[reportUnusedClass]
            ParametrizedAttribute, TypeAttribute
        ):
            name = "test_af.bad_unknown"
            value: IntegerType = param_def()
            assembly_format = "$nonexistent"


def test_error_assembly_format_with_parse_parameters():
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


def test_error_assembly_format_with_print_parameters():
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


def test_error_assembly_format_on_singleton():
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


# ============================================================================
# Directive tests: params, struct, qualified, custom, ref
# ============================================================================

# --- params directive attribute definitions ---


@irdl_attr_definition
class ParamsTwoType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.params_two"
    first: IntegerType = param_def()
    second: IntegerType = param_def()
    assembly_format = "params"


@irdl_attr_definition
class ParamsThreeType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.params_three"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    c: IntegerType = param_def()
    assembly_format = "params"


@irdl_attr_definition
class ParamsOptType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.params_opt"
    required: IntegerType = param_def()
    opt: NoneAttr | IntegerType = param_def()
    assembly_format = "params"


@irdl_attr_definition
class ParamsBracketedType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.params_brack"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    assembly_format = "`<` params `>`"


# --- struct directive attribute definitions ---


@irdl_attr_definition
class StructAllType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.struct_all"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    assembly_format = "struct(params)"


@irdl_attr_definition
class StructSelectType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.struct_sel"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    assembly_format = "struct($a, $b)"


@irdl_attr_definition
class StructOptType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.struct_opt"
    a: IntegerType = param_def()
    b: NoneAttr | IntegerType = param_def()
    assembly_format = "struct(params)"


# --- qualified directive attribute definitions ---


@irdl_attr_definition
class QualifiedParamType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.qual_param"
    value: IntegerType = param_def()
    assembly_format = "qualified($value)"


# --- custom directive class definitions ---


@irdl_attr_custom_directive
class PassthroughDir(AttrCustomDirective):
    param: ParameterVariable

    def parse(self, parser: AttrParser, state: AttrParsingState) -> bool:
        return self.param.parse(parser, state)

    def print(
        self, printer: Printer, state: PrintingState, attr: ParametrizedAttribute
    ) -> None:
        self.param.print(printer, state, attr)


@irdl_attr_custom_directive
class RefTestDir(AttrCustomDirective):
    first: ParameterVariable
    second_ref: ParameterVariable

    def parse(self, parser: AttrParser, state: AttrParsingState) -> bool:
        return self.first.parse(parser, state)

    def print(
        self, printer: Printer, state: PrintingState, attr: ParametrizedAttribute
    ) -> None:
        self.first.print(printer, state, attr)


# --- custom directive attribute definitions ---


@irdl_attr_definition
class CustomDirType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.custom_dir"
    value: IntegerType = param_def()
    assembly_format = "custom<PassthroughDir>($value)"
    custom_directives = (PassthroughDir,)


@irdl_attr_definition
class RefTestType(ParametrizedAttribute, TypeAttribute):
    name = "test_af.ref_test"
    a: IntegerType = param_def()
    b: IntegerType = param_def()
    assembly_format = "$b `,` custom<RefTestDir>($a, ref($b))"
    custom_directives = (RefTestDir,)


# --- Fixture ---


@pytest.fixture
def ctx_directives() -> Context:
    ctx = Context(allow_unregistered=True)
    for t in [
        ParamsTwoType,
        ParamsThreeType,
        ParamsOptType,
        ParamsBracketedType,
        StructAllType,
        StructSelectType,
        StructOptType,
        QualifiedParamType,
        CustomDirType,
        RefTestType,
    ]:
        ctx.load_attr_or_type(t)
    return ctx


# --- Helper ---


def _roundtrip_constructed(attr: ParametrizedAttribute, ctx: Context) -> None:
    """Print → parse → compare."""
    printed = print_attr(attr)
    if isinstance(attr, TypeAttribute):
        parsed = parse_type(printed, ctx)
    else:
        parsed = parse_attr(printed, ctx)
    assert attr == parsed, f"Round-trip failed: {printed!r}"


# ============================================================================
# params directive tests
# ============================================================================


def test_params_two_roundtrip(ctx_directives: Context):
    _roundtrip_constructed(ParamsTwoType(i32, i64), ctx_directives)


def test_params_three_roundtrip(ctx_directives: Context):
    _roundtrip_constructed(ParamsThreeType(i1, i32, i64), ctx_directives)


def test_params_optional_absent(ctx_directives: Context):
    attr = ParamsOptType(i32, NoneAttr())
    _roundtrip_constructed(attr, ctx_directives)
    printed = print_attr(attr)
    parsed = parse_type(printed, ctx_directives)
    assert isinstance(parsed, ParamsOptType)
    assert isinstance(parsed.opt, NoneAttr)


def test_params_optional_present(ctx_directives: Context):
    attr = ParamsOptType(i32, i64)
    _roundtrip_constructed(attr, ctx_directives)
    printed = print_attr(attr)
    parsed = parse_type(printed, ctx_directives)
    assert isinstance(parsed, ParamsOptType)
    assert parsed.opt == i64


def test_params_bracketed_roundtrip(ctx_directives: Context):
    _roundtrip_constructed(ParamsBracketedType(i32, i64), ctx_directives)


def test_params_bracketed_has_nested_brackets():
    printed = print_attr(ParamsBracketedType(i32, i64))
    assert printed.startswith("!test_af.params_brack<<")
    assert printed.endswith(">>")


# ============================================================================
# struct directive tests
# ============================================================================


def test_struct_all_roundtrip(ctx_directives: Context):
    _roundtrip_constructed(StructAllType(i32, i64), ctx_directives)


def test_struct_select_roundtrip(ctx_directives: Context):
    _roundtrip_constructed(StructSelectType(i32, i64), ctx_directives)


def test_struct_prints_key_value():
    printed = print_attr(StructAllType(i32, i64))
    assert "a =" in printed or "a=" in printed
    assert "b =" in printed or "b=" in printed


def test_struct_any_order(ctx_directives: Context):
    """Struct accepts parameters in any order during parsing."""
    forward = parse_type("!test_af.struct_all<a = i32, b = i64>", ctx_directives)
    reverse = parse_type("!test_af.struct_all<b = i64, a = i32>", ctx_directives)
    assert forward == reverse


def test_struct_optional_absent(ctx_directives: Context):
    attr = StructOptType(i32, NoneAttr())
    _roundtrip_constructed(attr, ctx_directives)
    printed = print_attr(attr)
    assert "b" not in printed


def test_struct_optional_present(ctx_directives: Context):
    _roundtrip_constructed(StructOptType(i32, i64), ctx_directives)


def test_struct_missing_required_param(ctx_directives: Context):
    with pytest.raises(ParseError, match="missing required parameter"):
        parse_type("!test_af.struct_all<b = i64>", ctx_directives)


# ============================================================================
# qualified directive tests
# ============================================================================


@pytest.mark.parametrize("value", [i32, i64])
def test_qualified_roundtrip(value: IntegerType, ctx_directives: Context):
    _roundtrip_constructed(QualifiedParamType(value), ctx_directives)


# ============================================================================
# custom directive tests
# ============================================================================


@pytest.mark.parametrize("value", [i32, i64])
def test_custom_directive_roundtrip(value: IntegerType, ctx_directives: Context):
    _roundtrip_constructed(CustomDirType(value), ctx_directives)


# ============================================================================
# ref directive tests
# ============================================================================


def test_ref_directive_roundtrip(ctx_directives: Context):
    _roundtrip_constructed(RefTestType(i32, i64), ctx_directives)
