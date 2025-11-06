from __future__ import annotations

import textwrap
from collections.abc import Callable
from io import StringIO
from typing import ClassVar, Generic

import pytest
from typing_extensions import TypeVar

from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import (
    I32,
    I64,
    BoolAttr,
    DenseArrayBase,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    StringAttr,
    SymbolNameConstraint,
    UnitAttr,
    f32,
    i32,
)
from xdsl.dialects.test import Test, TestType
from xdsl.dialects.utils import DynamicIndexList
from xdsl.ir import (
    Attribute,
    Operation,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyInt,
    AttrConstraint,
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    BaseAttr,
    EqAttrConstraint,
    IntVarConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    ParsePropInAttrDict,
    RangeOf,
    RangeVarConstraint,
    SameVariadicOperandSize,
    SameVariadicResultSize,
    TypedAttributeConstraint,  # pyright: ignore[reportDeprecated]
    VarConstraint,
    VarOperand,
    VarOpResult,
    attr_def,
    eq,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    opt_region_def,
    opt_result_def,
    opt_successor_def,
    prop_def,
    region_def,
    result_def,
    successor_def,
    var_operand_def,
    var_region_def,
    var_result_def,
    var_successor_def,
)
from xdsl.irdl.declarative_assembly_format import (
    AttrDictDirective,
    AttributeVariable,
    CustomDirective,
    FormatProgram,
    OperandsDirective,
    OperandVariable,
    ParsingState,
    PrintingState,
    PunctuationDirective,
    RegionVariable,
    ResultsDirective,
    SuccessorVariable,
    TypeDirective,
    VariadicOperandVariable,
    irdl_custom_directive,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import (
    ParseError,
    PyRDLError,
    PyRDLOpDefinitionError,
    VerifyException,
)

################################################################################
# Utils for this test file                                                     #
################################################################################


def check_roundtrip(program: str, ctx: Context):
    """Check that the given program roundtrips exactly (including whitespaces)."""
    parser = Parser(ctx, program)
    ops: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops.append(op)

    res_io = StringIO()
    printer = Printer(stream=res_io)
    for op in ops[:-1]:
        printer.print_op(op)
        printer.print_string("\n")
    printer.print_op(ops[-1])

    assert program == res_io.getvalue()


def check_equivalence(program1: str, program2: str, ctx: Context):
    """Check that the given programs are structurally equivalent."""

    parser = Parser(ctx, program1)
    ops1: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops1.append(op)

    parser = Parser(ctx, program2)
    ops2: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops2.append(op)

    mod1 = ModuleOp(ops1)
    mod2 = ModuleOp(ops2)

    assert mod1.is_structurally_equivalent(mod2), str(mod1) + "\n!=\n" + str(mod2)


################################################################################
# Tests that we cannot have both a declarative and Python-defined format       #
################################################################################


def test_format_and_print_op():
    """
    Check that an operation with an assembly format cannot redefine the print method.
    """
    with pytest.raises(
        PyRDLOpDefinitionError, match="Cannot define both an assembly format"
    ):

        @irdl_op_definition
        class FormatAndPrintOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.format_and_print"

            assembly_format = "attr-dict"

            def print(self, printer: Printer) -> None:
                pass


def test_format_and_parse_op():
    """
    Check that an operation with an assembly format cannot redefine the parse method.
    """
    with pytest.raises(
        PyRDLOpDefinitionError, match="Cannot define both an assembly format"
    ):

        @irdl_op_definition
        class FormatAndParseOp(IRDLOperation):
            name = "test.format_and_parse"

            assembly_format = "attr-dict"

            @classmethod
            def parse(cls, parser: Parser) -> FormatAndParseOp:
                raise NotImplementedError()


################################################################################
# 'attr-dict' directive                                                        #
################################################################################


def test_expected_attr_dict():
    """Check that an attr-dict directive is expected."""

    with pytest.raises(PyRDLOpDefinitionError, match="'attr-dict' directive not found"):

        @irdl_op_definition
        class NoAttrDictOp0(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_attr_dict"

            assembly_format = ""


def test_two_attr_dicts():
    """Check that we cannot have two attr-dicts."""

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive can only occur once"
    ):

        @irdl_op_definition
        class NoAttrDictOp1(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive can only occur once"
    ):

        @irdl_op_definition
        class NoAttrDictOp2(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict attr-dict-with-keyword"

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive can only occur once"
    ):

        @irdl_op_definition
        class NoAttrDictOp3(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict-with-keyword attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive can only occur once"
    ):

        @irdl_op_definition
        class NoAttrDictOp4(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict-with-keyword attr-dict-with-keyword"


@irdl_op_definition
class AttrDictOp(IRDLOperation):
    name = "test.attr_dict"

    assembly_format = "attr-dict"


@irdl_op_definition
class AttrDictWithKeywordOp(IRDLOperation):
    name = "test.attr_dict_with_keyword"

    assembly_format = "attr-dict-with-keyword"


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.attr_dict", '"test.attr_dict"() : () -> ()'),
        ("test.attr_dict_with_keyword", '"test.attr_dict_with_keyword"() : () -> ()'),
        (
            "test.attr_dict {a = 2 : i32}",
            '"test.attr_dict"() {a = 2 : i32} : () -> ()',
        ),
        (
            "test.attr_dict_with_keyword attributes {a = 2 : i32}",
            '"test.attr_dict_with_keyword"() {a = 2 : i32} : () -> ()',
        ),
    ],
)
def test_attr_dict(program: str, generic_program: str):
    """Test the 'attr-dict' and 'attr-dict-with-keyword' directives."""
    ctx = Context()
    ctx.load_op(AttrDictOp)
    ctx.load_op(AttrDictWithKeywordOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.prop {prop = true}", '"test.prop"() <{prop = true}> : () -> ()'),
        (
            "test.prop {a = 2 : i32, prop = true}",
            '"test.prop"() <{prop = true}> {a = 2 : i32} : () -> ()',
        ),
    ],
)
def test_attr_dict_prop_fallback(program: str, generic_program: str):
    @irdl_op_definition
    class PropOp(IRDLOperation):
        name = "test.prop"
        prop = opt_prop_def()
        irdl_options = [ParsePropInAttrDict()]
        assembly_format = "attr-dict"

    ctx = Context()
    ctx.load_op(PropOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.prop false {prop2 = true}",
            '"test.prop"() <{prop1 = false, prop2 = true}> : () -> ()',
        ),
        (
            "test.prop false {a = 2 : i32, prop2 = true}",
            '"test.prop"() <{prop1 = false, prop2 = true}> {a = 2 : i32} : () -> ()',
        ),
    ],
)
def test_partial_attr_dict_prop_fallback(program: str, generic_program: str):
    @irdl_op_definition
    class PropOp(IRDLOperation):
        name = "test.prop"
        prop1 = prop_def()
        prop2 = opt_prop_def()
        irdl_options = [ParsePropInAttrDict()]
        assembly_format = "$prop1 attr-dict"

    ctx = Context()
    ctx.load_op(PropOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


################################################################################
# Attribute variables                                                          #
################################################################################


@irdl_op_definition
class OpWithAttrOp(IRDLOperation):
    name = "test.one_attr"

    attr = attr_def()
    assembly_format = "$attr attr-dict"


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_attr i32", '"test.one_attr"() {attr = i32} : () -> ()'),
        (
            "test.one_attr i32 {attr2 = i64}",
            '"test.one_attr"() {attr = i32, attr2 = i64} : () -> ()',
        ),
    ],
)
def test_standard_attr_directive(program: str, generic_program: str):
    ctx = Context()
    ctx.load_op(OpWithAttrOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


def test_attr_variable_shadowed():
    ctx = Context()
    ctx.load_op(OpWithAttrOp)

    parser = Parser(ctx, "test.one_attr i32 {attr = i64}")
    with pytest.raises(
        ParseError,
        match="attributes attr are defined in other parts",
    ):
        parser.parse_operation()


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_attr i32", '"test.one_attr"() {irdl = i32} : () -> ()'),
        (
            "test.one_attr i32 {attr2 = i64}",
            '"test.one_attr"() {irdl = i32, attr2 = i64} : () -> ()',
        ),
    ],
)
def test_attr_name(program: str, generic_program: str):
    @irdl_op_definition
    class RenamedAttrOp(IRDLOperation):
        name = "test.one_attr"

        python = attr_def(Attribute, attr_name="irdl")
        assembly_format = "$irdl attr-dict"

    ctx = Context()
    ctx.load_op(RenamedAttrOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.one_attr <5 : i64>",
            '"test.one_attr"() {attr = #test.param<5 : i64>} : () -> ()',
        ),
        (
            'test.one_attr <"hello">',
            '"test.one_attr"() {attr = #test.param<"hello">} : () -> ()',
        ),
        (
            'test.one_attr <#test.param<"nested">>',
            '"test.one_attr"() {attr = #test.param<#test.param<"nested">>} : () -> ()',
        ),
    ],
)
def test_unqualified_attr(program: str, generic_program: str):
    @irdl_attr_definition
    class ParamOne(ParametrizedAttribute):
        name = "test.param"
        p: Attribute

    @irdl_op_definition
    class UnqualifiedAttrOp(IRDLOperation):
        name = "test.one_attr"

        attr = attr_def(ParamOne)
        assembly_format = "$attr attr-dict"

    ctx = Context()
    ctx.load_attr_or_type(ParamOne)
    ctx.load_op(UnqualifiedAttrOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


def test_missing_property_error():
    class MissingPropOp(IRDLOperation):
        name = "test.missing_prop"

        prop1 = prop_def()
        prop2 = prop_def()
        assembly_format = "$prop1 attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="prop2 properties are missing",
    ):
        irdl_op_definition(MissingPropOp)


def test_attribute_duplicated():
    """Test that attributes should not be parsed twice"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="attribute 'attr' is already bound"
    ):

        @irdl_op_definition
        class DuplicatedAttributeOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_attribute_op"
            attr = attr_def()

            assembly_format = "$attr $attr attr-dict"


def test_property_duplicated():
    """Test that properties should not be parsed twice"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="property 'attr' is already bound"
    ):

        @irdl_op_definition
        class DuplicatedPropertiesOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_property_op"
            attr = prop_def()

            assembly_format = "$attr $attr attr-dict"


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_prop i32", '"test.one_prop"() <{prop = i32}> : () -> ()'),
        (
            "test.one_prop i32 {attr2 = i64}",
            '"test.one_prop"() <{prop = i32}> {attr2 = i64} : () -> ()',
        ),
    ],
)
def test_standard_prop_directive(program: str, generic_program: str):
    @irdl_op_definition
    class PropOp(IRDLOperation):
        name = "test.one_prop"

        prop = prop_def()
        assembly_format = "$prop attr-dict"

    ctx = Context()
    ctx.load_op(PropOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_prop i32", '"test.one_prop"() <{irdl = i32}> : () -> ()'),
        (
            "test.one_prop i32 {attr2 = i64}",
            '"test.one_prop"() <{irdl = i32}> {attr2 = i64} : () -> ()',
        ),
    ],
)
def test_prop_name(program: str, generic_program: str):
    @irdl_op_definition
    class RenamedPropOp(IRDLOperation):
        name = "test.one_prop"

        python = prop_def(Attribute, prop_name="irdl")
        assembly_format = "$irdl attr-dict"

    ctx = Context()
    ctx.load_op(RenamedPropOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_property",
            '"test.optional_property"() : () -> ()',
        ),
        (
            "test.optional_property prop i32",
            '"test.optional_property"() <{prop = i32}> : () -> ()',
        ),
    ],
)
def test_optional_property(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalPropertyOp(IRDLOperation):
        name = "test.optional_property"
        prop = opt_prop_def()

        assembly_format = "(`prop` $prop^)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalPropertyOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_property",
            '"test.optional_property"() : () -> ()',
        ),
        (
            "test.optional_property i32",
            '"test.optional_property"() <{prop = i32}> : () -> ()',
        ),
    ],
)
def test_optional_qualified_property(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalPropertyOp(IRDLOperation):
        name = "test.optional_property"
        prop = opt_prop_def()

        assembly_format = "($prop^)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalPropertyOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_property()",
            '"test.optional_property"() : () -> ()',
        ),
        (
            "test.optional_property( prop i32 )",
            '"test.optional_property"() <{prop = i32}> : () -> ()',
        ),
    ],
)
def test_optional_property_with_whitespace(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalPropertyOp(IRDLOperation):
        name = "test.optional_property"
        prop = opt_prop_def()

        assembly_format = "`(` (` ` `prop` $prop^ ` `)? `)` attr-dict"

    ctx = Context()
    ctx.load_op(OptionalPropertyOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_unit_attr_prop",
            '"test.optional_unit_attr_prop"() : () -> ()',
        ),
        (
            "test.optional_unit_attr_prop unit_attr",
            '"test.optional_unit_attr_prop"() <{unit_attr}> : () -> ()',
        ),
    ],
)
def test_optional_unit_attr_property(program: str, generic_program: str):
    """Test the parsing of optional UnitAttr operands"""

    @irdl_op_definition
    class OptionalUnitAttrPropertyOp(IRDLOperation):
        name = "test.optional_unit_attr_prop"
        unit_attr = opt_prop_def(UnitAttr)

        assembly_format = "(`unit_attr` $unit_attr^)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalUnitAttrPropertyOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_unit_attr",
            '"test.optional_unit_attr"() : () -> ()',
        ),
        (
            "test.optional_unit_attr unit_attr",
            '"test.optional_unit_attr"() <{unit_attr}> : () -> ()',
        ),
    ],
)
def test_optional_unit_attr_attribute(program: str, generic_program: str):
    """Test the parsing of optional UnitAttr operands"""

    @irdl_op_definition
    class OptionalUnitAttrOp(IRDLOperation):
        name = "test.optional_unit_attr"
        unit_attr = opt_prop_def(UnitAttr)

        assembly_format = "(`unit_attr` $unit_attr^)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalUnitAttrOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_attribute",
            '"test.optional_attribute"() : () -> ()',
        ),
        (
            "test.optional_attribute attr i32",
            '"test.optional_attribute"() {attr = i32} : () -> ()',
        ),
    ],
)
def test_optional_attribute(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalAttributeOp(IRDLOperation):
        name = "test.optional_attribute"
        attr = opt_attr_def()

        assembly_format = "(`attr` $attr^)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalAttributeOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.typed_attr 3 3.000000e+00",
            '"test.typed_attr"() {attr = 3 : i32, float_attr = 3.000000e+00 : f64} : () -> ()',
        ),
    ],
)
def test_typed_attribute_variable(program: str, generic_program: str):
    """Test the parsing of typed attributes"""

    @irdl_op_definition
    class TypedAttributeOp(IRDLOperation):
        name = "test.typed_attr"
        attr = attr_def(IntegerAttr[I32])
        float_attr = attr_def(FloatAttr[Float64Type])

        assembly_format = "$attr $float_attr attr-dict"

    ctx = Context()
    ctx.load_op(TypedAttributeOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.symbol @test @test2",
            '"test.symbol"() <{sym_name = "test"}> {attr_sym_name = "test2"} : () -> ()',
        ),
        (
            'test.symbol @"123" @"456"',
            '"test.symbol"() <{sym_name = "123"}> {attr_sym_name = "456"} : () -> ()',
        ),
    ],
)
def test_symbol_name_variable(program: str, generic_program: str):
    @irdl_op_definition
    class SymbolNameOp(IRDLOperation):
        name = "test.symbol"

        sym_name = prop_def(SymbolNameConstraint())
        attr_sym_name = attr_def(SymbolNameConstraint())

        assembly_format = "$sym_name $attr_sym_name attr-dict"

    ctx = Context()
    ctx.load_op(SymbolNameOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.symbol @test symbol @test2",
            '"test.symbol"() <{sym_name = "test2", other_sym_name = "test"}> : () -> ()',
        ),
        (
            "test.symbol",
            '"test.symbol"() : () -> ()',
        ),
    ],
)
def test_optional_symbol_name_variable(program: str, generic_program: str):
    @irdl_op_definition
    class SymbolNameOp(IRDLOperation):
        name = "test.symbol"

        sym_name = opt_prop_def(SymbolNameConstraint())

        other_sym_name = opt_prop_def(SymbolNameConstraint())

        assembly_format = "$other_sym_name (`symbol` $sym_name^)? attr-dict"

    ctx = Context()
    ctx.load_op(SymbolNameOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program, format",
    [
        (
            "test.symbol [1, 2] [3, 4] [5.000000e+00, 6.000000e+00] [7.000000e+00]",
            '"test.symbol"() <{i64s = array<i64: 3, 4>, f32s = array<f32: 5.000000e+00, 6.000000e+00>, f64s = array<f64: 7.000000e+00>}> {i32s = array<i32: 1, 2>} : () -> ()',
            "$i32s $i64s $f32s $f64s attr-dict",
        ),
        (
            "test.symbol [7, 8] [1.000000e+01, 1.100000e+01]",
            '"test.symbol"() <{f32s = array<f32: 9.000000e+00>, f64s = array<f64: 1.000000e+01, 1.100000e+01>}> {i32s = array<i32: 7, 8>} : () -> ()',
            "$i32s (`i64s` $i64s^)? (`f32s` $f32s^)? $f64s attr-dict",
        ),
    ],
)
def test_dense_array_special_cases(program: str, generic_program: str, format: str):
    @irdl_op_definition
    class DenseArrayOp(IRDLOperation):
        name = "test.symbol"

        i32s = attr_def(DenseArrayBase[I32])
        i64s = opt_prop_def(DenseArrayBase[I64])
        f32s = prop_def(
            DenseArrayBase[Float32Type],
            default_value=DenseArrayBase.from_list(f32, (9.0,)),
        )
        f64s = prop_def(
            VarConstraint("F64S", irdl_to_attr_constraint(DenseArrayBase[Float64Type]))
        )

        assembly_format = format

    ctx = Context()
    ctx.load_op(DenseArrayOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


################################################################################
# Punctuations, keywords, and whitespaces                                      #
################################################################################


@pytest.mark.parametrize(
    "format, program",
    [
        ("`)` attr-dict", "test.punctuation)"),
        ("`(` attr-dict", "test.punctuation("),
        ("`->` attr-dict", "test.punctuation ->"),
        (
            "`->` `->` attr-dict",
            "test.punctuation -> ->",
        ),
        (
            "`(` `)` attr-dict",
            "test.punctuation()",
        ),
        (
            "`keyword` attr-dict",
            "test.punctuation keyword",
        ),
        (
            "`keyword` `,` `keyword` attr-dict",
            "test.punctuation keyword, keyword",
        ),
        ("`keyword` ` ` `,` `keyword` attr-dict", "test.punctuation keyword , keyword"),
        ("`keyword` `,` `` `keyword` attr-dict", "test.punctuation keyword,keyword"),
        (
            "`keyword` `\\n` `,` `keyword` attr-dict",
            "test.punctuation keyword\n, keyword",
        ),
        (
            "`keyword` `,` ` ` `keyword` attr-dict",
            "test.punctuation keyword, keyword",
        ),
        (
            "`keyword` `,` ` ` ` ` `keyword` attr-dict",
            "test.punctuation keyword,  keyword",
        ),
    ],
)
def test_punctuations_and_keywords(format: str, program: str):
    """Test the punctuation and keyword directives."""

    @irdl_op_definition
    class PunctuationOp(IRDLOperation):
        name = "test.punctuation"
        assembly_format = format

    ctx = Context()
    ctx.load_op(PunctuationOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, '"test.punctuation"() : () -> ()', ctx)


################################################################################
# Variables                                                                    #
################################################################################


def test_unknown_variable():
    """Test that variables should refer to an element in the operation."""
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="expected variable to refer to an operand, attribute, region, or successor",
    ) as exc_info:

        @irdl_op_definition
        class UnknownVarOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.unknown_var_op"

            assembly_format = "$var attr-dict"

    assert isinstance(exc_info.value.__cause__, ParseError)
    assert exc_info.value.__cause__.span.text == "$var"


################################################################################
# Operands                                                                     #
################################################################################


def test_missing_operand():
    """Test that operands should have their type parsed."""
    with pytest.raises(PyRDLOpDefinitionError, match="operand 'operand' not found"):

        @irdl_op_definition
        class NoOperandTypeOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_operand_type_op"
            operand = operand_def()

            assembly_format = "attr-dict"


def test_operands_missing_type():
    """Test that operands should have their type parsed"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="type of operand 'operand' cannot be inferred"
    ):

        @irdl_op_definition
        class NoOperandTypeOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_operand_type_op"
            operand = operand_def()

            assembly_format = "$operand attr-dict"


def test_operands_duplicated():
    """Test that operands should not be parsed twice"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="operand 'operand' is already bound"
    ):

        @irdl_op_definition
        class DuplicatedOperandOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_operand_op"
            operand = operand_def()

            assembly_format = "$operand $operand type($operand) attr-dict"


def test_operands_duplicated_type():
    """Test that operands should not have their type parsed twice"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="type of 'operand' is already bound"
    ):

        @irdl_op_definition
        class DuplicatedOperandTypeOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_operand_type_op"
            operand = operand_def()

            assembly_format = "$operand type($operand) type($operand) attr-dict"


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "$lhs $rhs type($lhs) type($rhs) attr-dict",
            '%0, %1 = "test.op"() : () -> (i32, i64)\ntest.two_operands %0 %1 i32 i64',
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.two_operands"(%0, %1) : (i32, i64) -> ()',
        ),
        (
            "$rhs $lhs type($rhs) type($lhs) attr-dict",
            '%0, %1 = "test.op"() : () -> (i32, i64)\ntest.two_operands %1 %0 i64 i32',
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.two_operands"(%0, %1) : (i32, i64) -> ()',
        ),
        (
            "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.two_operands %0, %1 : i32, i64",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.two_operands"(%0, %1) : (i32, i64) -> ()',
        ),
    ],
)
def test_operands(format: str, program: str, generic_program: str):
    """Test the parsing of operands"""

    @irdl_op_definition
    class TwoOperandsOp(IRDLOperation):
        name = "test.two_operands"
        lhs = operand_def()
        rhs = operand_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(TwoOperandsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\ntest.variadic_operand',
            '%0 = "test.op"() : () -> i32\n"test.variadic_operand"() : () -> ()',
        ),
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\ntest.variadic_operand %0 i32',
            '%0 = "test.op"() : () -> i32\n"test.variadic_operand"(%0) : (i32) -> ()',
        ),
        (
            "$args type($args) attr-dict",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.variadic_operand %0, %1 i32, i64",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.variadic_operand"(%0, %1) : (i32, i64) -> ()',
        ),
        (
            "$args `:` type($args) attr-dict",
            '%0, %1, %2 = "test.op"() : () -> (i32, i64, i128)\n'
            "test.variadic_operand %0, %1, %2 : i32, i64, i128",
            '%0, %1, %2 = "test.op"() : () -> (i32, i64, i128)\n'
            '"test.variadic_operand"(%0, %1, %2) : (i32, i64, i128) -> ()',
        ),
    ],
)
def test_variadic_operand(format: str, program: str, generic_program: str):
    """Test the parsing of variadic operands"""

    @irdl_op_definition
    class VariadicOperandOp(IRDLOperation):
        name = "test.variadic_operand"
        args = var_operand_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(VariadicOperandOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\ntest.optional_operand',
            '%0 = "test.op"() : () -> i32\n"test.optional_operand"() : () -> ()',
        ),
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\ntest.optional_operand %0 i32',
            '%0 = "test.op"() : () -> i32\n"test.optional_operand"(%0) : (i32) -> ()',
        ),
    ],
)
def test_optional_operand(format: str, program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalOperandOp(IRDLOperation):
        name = "test.optional_operand"
        args = opt_operand_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(OptionalOperandOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program, as_property",
    [
        (
            '%0 = "test.op"() : () -> i32\ntest.variadic_operands(%0 : i32) [%0 : i32]',
            '%0 = "test.op"() : () -> i32\n'
            '"test.variadic_operands"(%0, %0) {operandSegmentSizes = array<i32:1,1>} : (i32,i32) -> ()',
            False,
        ),
        (
            '%0 = "test.op"() : () -> i32\ntest.variadic_operands(%0 : i32) [%0 : i32]',
            '%0 = "test.op"() : () -> i32\n'
            '"test.variadic_operands"(%0, %0) <{operandSegmentSizes = array<i32:1,1>}> : (i32,i32) -> ()',
            True,
        ),
        (
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.variadic_operands(%0, %1 : i32, i64) [%1, %0 : i64, i32]",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.variadic_operands"(%0, %1, %1, %0) {operandSegmentSizes = array<i32:2,2>} : (i32, i64, i64, i32) -> ()',
            False,
        ),
        (
            '%0, %1, %2 = "test.op"() : () -> (i32, i64, i128)\n'
            "test.variadic_operands(%0, %1, %2 : i32, i64, i128) [%2, %1, %0 : i128, i64, i32]",
            '%0, %1, %2 = "test.op"() : () -> (i32, i64, i128)\n'
            '"test.variadic_operands"(%0, %1, %2, %2, %1, %0) {operandSegmentSizes = array<i32:3,3>} : (i32, i64, i128, i128, i64, i32) -> ()',
            False,
        ),
    ],
)
def test_multiple_variadic_operands(
    program: str, generic_program: str, as_property: bool
):
    """Test the parsing of variadic operands"""

    @irdl_op_definition
    class VariadicOperandsOp(IRDLOperation):
        name = "test.variadic_operands"
        args1 = var_operand_def()
        args2 = var_operand_def()

        irdl_options = [AttrSizedOperandSegments(as_property=as_property)]

        assembly_format = (
            "`(` $args1 `:` type($args1) `)` `[` $args2 `:` type($args2) `]` attr-dict"
        )

    ctx = Context()
    ctx.load_op(VariadicOperandsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_operands(:) [:]",
            '"test.optional_operands"() {operandSegmentSizes = array<i32:0,0>} : () -> ()',
        ),
        (
            '%0 = "test.op"() : () -> i32\ntest.optional_operands(%0 : i32) [%0 : i32]',
            '%0 = "test.op"() : () -> i32\n'
            '"test.optional_operands"(%0, %0) {operandSegmentSizes = array<i32:1,1>} : (i32,i32) -> ()',
        ),
    ],
)
def test_multiple_optional_operands(program: str, generic_program: str):
    """Test the parsing of variadic operands"""

    @irdl_op_definition
    class OptionalOperandsOp(IRDLOperation):
        name = "test.optional_operands"
        arg1 = opt_operand_def()
        arg2 = opt_operand_def()

        irdl_options = [AttrSizedOperandSegments()]

        assembly_format = (
            "`(` $arg1 `:` type($arg1) `)` `[` $arg2 `:` type($arg2) `]` attr-dict"
        )

    ctx = Context()
    ctx.load_op(OptionalOperandsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program",
    [
        (
            "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict",
            "test.two_operands %0, %1 : i32, i64\n"
            '%0, %1 = "test.op"() : () -> (i32, i64)',
        ),
    ],
)
def test_operands_graph_region(format: str, program: str):
    """Test the parsing of operands in a graph region"""

    @irdl_op_definition
    class TwoOperandsOp(IRDLOperation):
        name = "test.two_operands"
        lhs = operand_def()
        rhs = operand_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(TwoOperandsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "test.operands_directive %0 : i32",
        "test.operands_directive %0, %1 : i32, i32",
        "test.operands_directive %0, %1, %2 : i32, i32, i32",
    ],
)
def test_operands_directive_with_variadic(program: str):
    """Test the operands directive with a variadic operand"""

    @irdl_op_definition
    class OperandsDirectiveOp(IRDLOperation):
        name = "test.operands_directive"

        op1 = operand_def()
        op2 = var_operand_def()

        assembly_format = "operands `:` type(operands) attr-dict"

    ctx = Context()
    ctx.load_op(OperandsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "test.operands_directive %0 : i32",
        "test.operands_directive %0, %1 : i32, i32",
    ],
)
def test_operands_directive_with_optional(program: str):
    """Test the operands directive with an optional operand"""

    @irdl_op_definition
    class OperandsDirectiveOp(IRDLOperation):
        name = "test.operands_directive"

        op1 = opt_operand_def()
        op2 = operand_def()

        assembly_format = "operands `:` type(operands) attr-dict"

    ctx = Context()
    ctx.load_op(OperandsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


def test_operands_directive_with_no_variadic():
    """Test the operands directive with no variadic operands"""

    @irdl_op_definition
    class OperandsDirectiveOp(IRDLOperation):
        name = "test.operands_directive"

        op1 = operand_def()
        op2 = operand_def()

        assembly_format = "operands `:` type(operands) attr-dict"

    ctx = Context()
    ctx.load_op(OperandsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip("test.operands_directive %0, %1 : i32, i32", ctx)


def test_operands_directive_fails_with_two_var():
    """Test operands directive cannot be used with two variadic operands"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'operands' is ambiguous with multiple variadic operands",
    ):

        @irdl_op_definition
        class TwoVarOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.two_var_op"

            op1 = var_operand_def()
            op2 = var_operand_def()

            irdl_options = [AttrSizedOperandSegments()]

            assembly_format = "operands attr-dict `:` type(operands)"


@pytest.mark.parametrize(
    "program",
    [
        "test.two_var_op :",
        "test.two_var_op %0, %1 : i32, i32",
        "test.two_var_op %0, %1, %2, %3 : i32, i32, i32, i32",
    ],
)
def test_operands_directive_works_with_two_var_and_option(program: str):
    """
    Test operands directive can be used with two variadic operands as long as they have
    the same length.
    """

    @irdl_op_definition
    class TwoVarOp(IRDLOperation):
        name = "test.two_var_op"

        res1 = var_operand_def()
        res2 = var_operand_def()

        irdl_options = [SameVariadicOperandSize()]

        assembly_format = "operands attr-dict  `:` type(operands)"

    ctx = Context()
    ctx.load_op(TwoVarOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "test.two_var_op :",
        "test.two_var_op %0, %1 : i32, i32",
    ],
)
def test_operands_directive_works_with_two_opt_and_option(program: str):
    """
    Test operands directive can be used with two optional operands as long as they have
    the same length.
    """

    @irdl_op_definition
    class TwoVarOp(IRDLOperation):
        name = "test.two_var_op"

        res1 = var_operand_def()
        res2 = var_operand_def()

        irdl_options = [SameVariadicOperandSize()]

        assembly_format = "operands attr-dict `:` type(operands)"

    ctx = Context()
    ctx.load_op(TwoVarOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


def test_operands_directive_fails_with_no_operands():
    """Test operands directive cannot be used with no operands"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'operands' should not be used when there are no operands",
    ):

        @irdl_op_definition
        class NoOperandsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_operands_op"

            assembly_format = "operands attr-dict `:` type(operands)"


def test_operands_directive_fails_with_other_directive():
    """Test operands directive cannot be used with no operands"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'operands' cannot be used with other operand directives",
    ):

        @irdl_op_definition
        class TwoOperandsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.two_operands_op"

            op1 = operand_def()
            op2 = operand_def()

            assembly_format = "$op1 `,` operands attr-dict `:` type(operands)"


def test_operands_directive_fails_with_other_type_directive():
    """Test operands directive cannot be used with no operands"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'operands' cannot be used in a type directive with other operand type directives",
    ):

        @irdl_op_definition
        class TwoOperandsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.two_operands_op"

            op1 = operand_def()
            op2 = operand_def()

            assembly_format = "operands attr-dict `:` type($op1) `,` type(operands)"


@pytest.mark.parametrize(
    "program, error",
    [
        ("test.two_operands %0 : i32, i32", "Expected 2 operands, but got 1"),
        (
            "test.two_operands %0, %1, %2 : i32, i32",
            "Expected 2 operands, but got 3",
        ),
        ("test.two_operands %0, %1 : i32", "Expected 2 operand types, but got 1"),
        (
            "test.two_operands %0, %1 : i32, i32, i32",
            "Expected 2 operand types, but got 3",
        ),
    ],
)
def test_operands_directive_bounds(program: str, error: str):
    @irdl_op_definition
    class TwoOperandsOp(IRDLOperation):
        name = "test.two_operands"

        op1 = operand_def()
        op2 = operand_def()

        assembly_format = "operands attr-dict `:` type(operands)"

    ctx = Context()
    ctx.load_op(TwoOperandsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "test.three_operands %0 : i32, i32",
            "Expected 2 or 3 operands, but got 1",
        ),
        (
            "test.three_operands %0, %1, %2, %3 : i32, i32, i32",
            "Expected 2 or 3 operands, but got 4",
        ),
        (
            "test.three_operands %0, %1 : i32",
            "Expected 2 or 3 operand types, but got 1",
        ),
        (
            "test.three_operands %0, %1, %3 : i32, i32, i32, i32",
            "Expected 2 or 3 operand types, but got 4",
        ),
    ],
)
def test_operands_directive_bounds_with_opt(program: str, error: str):
    @irdl_op_definition
    class ThreeOperandsOp(IRDLOperation):
        name = "test.three_operands"

        op1 = operand_def()
        op2 = opt_operand_def()
        op3 = operand_def()

        assembly_format = "operands attr-dict `:` type(operands)"

    ctx = Context()
    ctx.load_op(ThreeOperandsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "test.three_operands %0 : i32, i32",
            "Expected at least 2 operands, but got 1",
        ),
        (
            "test.three_operands %0, %1 : i32",
            "Expected at least 2 operand types, but got 1",
        ),
    ],
)
def test_operands_directive_bound_with_var(program: str, error: str):
    @irdl_op_definition
    class ThreeOperandsOp(IRDLOperation):
        name = "test.three_operands"

        op1 = operand_def()
        op2 = var_operand_def()
        op3 = operand_def()

        assembly_format = "operands attr-dict `:` type(operands)"

    ctx = Context()
    ctx.load_op(ThreeOperandsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


def test_operands_directive_with_non_variadic_type_directive():
    """Tests the 'parse_single_type' function of the operands directive."""

    # The parser will never generate a non-variadic TypeDirective containing
    # an OperandsDirective, but we can manually make one.
    format_program = FormatProgram(
        (
            OperandsDirective(),
            AttrDictDirective(False, set(), set()),
            PunctuationDirective(":"),
            TypeDirective(OperandsDirective()),
        ),
    )

    @irdl_op_definition
    class OneOperandOp(IRDLOperation):
        name = "test.one_operand"

        op1 = operand_def()

        @classmethod
        def parse(cls, parser: Parser) -> OneOperandOp:
            return format_program.parse(parser, cls)

        def print(self, printer: Printer):
            format_program.print(printer, self)

    ctx = Context()
    ctx.load_op(OneOperandOp)

    check_roundtrip("test.one_operand %0 : i32", ctx)


def test_operands_directive_with_variadic_type_directive():
    """
    Tests the 'parse_single_type' function of the operands directive
    when the operation has a variadic.
    """
    # The parser will never generate a non-variadic TypeDirective containing
    # an OperandsDirective, but we can manually make one.
    format_program = FormatProgram(
        (
            OperandsDirective(),
            AttrDictDirective(False, set(), set()),
            PunctuationDirective(":"),
            TypeDirective(OperandsDirective()),
        ),
    )

    @irdl_op_definition
    class TwoOperandOp(IRDLOperation):
        name = "test.two_operand"

        op1 = operand_def()
        op2 = var_operand_def()

        @classmethod
        def parse(cls, parser: Parser) -> TwoOperandOp:
            return format_program.parse(parser, cls)

        def print(self, printer: Printer):
            format_program.print(printer, self)

    ctx = Context()
    ctx.load_op(TwoOperandOp)

    check_roundtrip("test.two_operand %0 : i32", ctx)


################################################################################
# Results                                                                      #
################################################################################


def test_missing_result_type():
    """Test that results should have their type parsed."""
    with pytest.raises(
        PyRDLOpDefinitionError, match="result 'result' cannot be inferred"
    ):

        @irdl_op_definition
        class NoResultTypeOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_result_type_op"
            result = result_def()

            assembly_format = "attr-dict"


def test_results_duplicated_type():
    """Test that results should not have their type parsed twice"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="type of 'result' is already bound"
    ):

        @irdl_op_definition
        class DuplicatedresultTypeOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_result_type_op"
            result = result_def()

            assembly_format = "type($result) type($result) attr-dict"


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "type($lhs) type($rhs) attr-dict",
            "%0, %1 = test.two_results i32 i64",
            '%0, %1 = "test.two_results"() : () -> (i32, i64)',
        ),
        (
            "type($rhs) type($lhs) attr-dict",
            "%0, %1 = test.two_results i32 i64",
            '%0, %1 = "test.two_results"() : () -> (i64, i32)',
        ),
        (
            "`:` type($lhs) `,` type($rhs) attr-dict",
            "%0, %1 = test.two_results : i32, i64",
            '%0, %1 = "test.two_results"() : () -> (i32, i64)',
        ),
    ],
)
def test_results(format: str, program: str, generic_program: str):
    """Test the parsing of results"""

    @irdl_op_definition
    class TwoResultOp(IRDLOperation):
        name = "test.two_results"
        lhs = result_def()
        rhs = result_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(TwoResultOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "`:` type($res) attr-dict",
            "test.variadic_result :",
            '"test.variadic_result"() : () -> ()',
        ),
        (
            "`:` type($res) attr-dict",
            "%0 = test.variadic_result : i32",
            '%0 = "test.variadic_result"() : () -> i32',
        ),
        (
            "`:` type($res) attr-dict",
            "%0, %1 = test.variadic_result : i32, i64",
            '%0, %1 = "test.variadic_result"() : () -> (i32, i64)',
        ),
        (
            "`:` type($res) attr-dict",
            "%0, %1, %2 = test.variadic_result : i32, i64, i128",
            '%0, %1, %2 = "test.variadic_result"() : () -> (i32, i64, i128)',
        ),
    ],
)
def test_variadic_result(format: str, program: str, generic_program: str):
    """Test the parsing of variadic results"""

    @irdl_op_definition
    class VariadicResultOp(IRDLOperation):
        name = "test.variadic_result"
        res = var_result_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(VariadicResultOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


def test_variadic_result_failure():
    """Test that inferring a range of inferrable attributes of unknown length fails."""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="type of result 'res' cannot be inferred",
    ):

        @irdl_op_definition
        class VariadicResultsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.var_results_op"

            res = var_result_def(IndexType())

            irdl_options = [AttrSizedResultSegments()]

            assembly_format = "attr-dict"


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "`:` type($res) attr-dict",
            "test.optional_result :",
            '"test.optional_result"() : () -> ()',
        ),
        (
            "`:` type($res) attr-dict",
            "%0 = test.optional_result : i32",
            '%0 = "test.optional_result"() : () -> i32',
        ),
    ],
)
def test_optional_result(format: str, program: str, generic_program: str):
    """Test the parsing of variadic results"""

    @irdl_op_definition
    class OptionalResultOp(IRDLOperation):
        name = "test.optional_result"
        res = opt_result_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(OptionalResultOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "%0 = test.results_directive : i32",
        "%0, %1 = test.results_directive : i32, i32",
        "%0, %1, %2 = test.results_directive : i32, i32, i32",
    ],
)
def test_results_directive_with_variadic(program: str):
    """Test the results directive with a variadic result"""

    @irdl_op_definition
    class ResultsDirectiveOp(IRDLOperation):
        name = "test.results_directive"

        res1 = result_def()
        res2 = var_result_def()

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(ResultsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "%0 = test.results_directive : i32",
        "%0, %1 = test.results_directive : i32, i32",
    ],
)
def test_results_directive_with_optional(program: str):
    """Test the results directive with an optional result"""

    @irdl_op_definition
    class ResultsDirectiveOp(IRDLOperation):
        name = "test.results_directive"

        res1 = opt_result_def()
        res2 = result_def()

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(ResultsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


def test_results_directive_with_no_variadic():
    """Test the results directive with no variadic results"""

    @irdl_op_definition
    class ResultsDirectiveOp(IRDLOperation):
        name = "test.results_directive"

        res1 = result_def()
        res2 = result_def()

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(ResultsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip("%0, %1 = test.results_directive : i32, i32", ctx)


def test_results_directive_fails_with_two_var():
    """Test results directive cannot be used with two variadic results"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'results' is ambiguous with multiple variadic results",
    ):

        @irdl_op_definition
        class TwoVarOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.two_var_op"

            res1 = var_result_def()
            res2 = var_result_def()

            irdl_options = [AttrSizedResultSegments()]

            assembly_format = "attr-dict `:` type(results)"


@pytest.mark.parametrize(
    "program",
    [
        "test.two_var_op :",
        "%0, %1 = test.two_var_op : i32, i32",
        "%0, %1, %2, %3 = test.two_var_op : i32, i32, i32, i32",
    ],
)
def test_results_directive_works_with_two_var_and_option(program: str):
    """
    Test results directive can be used with two variadic results as long as they have
    the same length.
    """

    @irdl_op_definition
    class TwoVarOp(IRDLOperation):
        name = "test.two_var_op"

        res1 = var_result_def()
        res2 = var_result_def()

        irdl_options = [SameVariadicResultSize()]

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(TwoVarOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "test.two_var_op :",
        "%0, %1 = test.two_var_op : i32, i32",
    ],
)
def test_results_directive_works_with_two_opt_and_option(program: str):
    """
    Test results directive can be used with two optional results as long as they have
    the same length.
    """

    @irdl_op_definition
    class TwoVarOp(IRDLOperation):
        name = "test.two_var_op"

        res1 = var_result_def()
        res2 = var_result_def()

        irdl_options = [SameVariadicResultSize()]

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(TwoVarOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


def test_results_directive_fails_with_no_results():
    """Test results directive cannot be used with no results"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'results' should not be used when there are no results",
    ):

        @irdl_op_definition
        class NoResultsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_results_op"

            assembly_format = "attr-dict `:` type(results)"


def test_results_directive_fails_with_other_type_directive():
    """Test results directive cannot be used with no results"""

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="'results' cannot be used in a type directive with other result type directives",
    ):

        @irdl_op_definition
        class TwoResultsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.two_results_op"

            res1 = result_def()
            res2 = result_def()

            assembly_format = "attr-dict `:` type($res1) `,` type(results)"


@pytest.mark.parametrize(
    "program, error",
    [
        ("%0 = test.two_results : i32", "Expected 2 result types, but got 1"),
        (
            "%0, %1, %2 = test.two_results : i32, i32, i32",
            "Expected 2 result types, but got 3",
        ),
    ],
)
def test_results_directive_bounds(program: str, error: str):
    @irdl_op_definition
    class TwoResultsOp(IRDLOperation):
        name = "test.two_results"

        res1 = result_def()
        res2 = result_def()

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(TwoResultsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "%0 = test.three_results : i32",
            "Expected 2 or 3 result types, but got 1",
        ),
        (
            "%0, %1, %2, %3 = test.three_results : i32, i32, i32, i32",
            "Expected 2 or 3 result types, but got 4",
        ),
    ],
)
def test_results_directive_bounds_with_opt(program: str, error: str):
    @irdl_op_definition
    class ThreeResultsOp(IRDLOperation):
        name = "test.three_results"

        res1 = result_def()
        res2 = opt_result_def()
        res3 = result_def()

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(ThreeResultsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


def test_results_directive_bound_with_var():
    @irdl_op_definition
    class ThreeResultsOp(IRDLOperation):
        name = "test.three_results"

        res1 = result_def()
        res2 = var_result_def()
        res3 = result_def()

        assembly_format = "attr-dict `:` type(results)"

    ctx = Context()
    ctx.load_op(ThreeResultsOp)

    with pytest.raises(ParseError, match="Expected at least 2 result types, but got 1"):
        parser = Parser(ctx, "%0 = test.three_results : i32")
        parser.parse_operation()


def test_results_directive_with_non_variadic_type_directive():
    """Tests the 'parse_single_type' function of the results directive."""

    # The parser will never generate a non-variadic TypeDirective containing
    # a ResultsDirective, but we can manually make one.
    format_program = FormatProgram(
        (
            AttrDictDirective(False, set(), set()),
            PunctuationDirective(":"),
            TypeDirective(ResultsDirective()),
        ),
    )

    @irdl_op_definition
    class OneResultOp(IRDLOperation):
        name = "test.one_result"

        res = result_def()

        @classmethod
        def parse(cls, parser: Parser) -> OneResultOp:
            return format_program.parse(parser, cls)

        def print(self, printer: Printer):
            format_program.print(printer, self)

    ctx = Context()
    ctx.load_op(OneResultOp)

    check_roundtrip("%0 = test.one_result : i32", ctx)


def test_results_directive_with_variadic_type_directive():
    """
    Tests the 'parse_single_type' function of the results directive
    when the operation has a variadic.
    """
    # The parser will never generate a non-variadic TypeDirective containing
    # a ResultsDirective, but we can manually make one.
    format_program = FormatProgram(
        (
            AttrDictDirective(False, set(), set()),
            PunctuationDirective(":"),
            TypeDirective(ResultsDirective()),
        ),
    )

    @irdl_op_definition
    class TwoResultsOp(IRDLOperation):
        name = "test.two_results"

        res1 = result_def()
        res2 = var_result_def()

        @classmethod
        def parse(cls, parser: Parser) -> TwoResultsOp:
            return format_program.parse(parser, cls)

        def print(self, printer: Printer):
            format_program.print(printer, self)

    ctx = Context()
    ctx.load_op(TwoResultsOp)

    check_roundtrip("%0 = test.two_results : i32", ctx)


################################################################################
# Functional type                                                              #
################################################################################


@pytest.mark.parametrize(
    "program",
    [
        "%0 = test.functional_type %1, %2 : (i32, i32) -> i32",
        "test.functional_type %0, %1 : (i32, i32) -> ()",
        "%0, %1 = test.functional_type %2, %3 : (i32, i32) -> (i32, i32)",
        "%0 = test.functional_type %1 : (i32) -> i32",
        "%0 = test.functional_type : () -> i32",
    ],
)
def test_functional_type(program: str):
    """Test the parsing of the functional-type directive"""

    @irdl_op_definition
    class FunctionalTypeOp(IRDLOperation):
        name = "test.functional_type"

        ops = var_operand_def()
        res = var_result_def()

        assembly_format = "$ops attr-dict `:` functional-type($ops, $res)"

    ctx = Context()
    ctx.load_op(FunctionalTypeOp)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program",
    [
        "%0 = test.functional_type %1, %2 : (i32, i32) -> i32",
        "%0, %1 = test.functional_type %2, %3 : (i32, i32) -> (i32, i32)",
        "%0 = test.functional_type %1 : (i32) -> i32",
    ],
)
def test_functional_type_with_operands_and_results(program: str):
    """
    Test the parsing of the functional-type directive using the operands and
    results directives
    """

    @irdl_op_definition
    class FunctionalTypeOp(IRDLOperation):
        name = "test.functional_type"

        op1 = operand_def()
        ops2 = var_operand_def()
        res1 = var_result_def()
        res2 = result_def()

        assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    ctx = Context()
    ctx.load_op(FunctionalTypeOp)

    check_roundtrip(program, ctx)


################################################################################
# Regions                                                                     #
################################################################################


def test_missing_region():
    """Test that regions should be parsed."""
    with pytest.raises(PyRDLOpDefinitionError, match="region 'region' not found"):

        @irdl_op_definition
        class NoRegionOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_region_op"
            region = region_def()

            assembly_format = "attr-dict-with-keyword"


def test_region_duplicated():
    """Test that regions should not be parsed twice"""
    with pytest.raises(PyRDLOpDefinitionError, match="region 'r' is already bound"):

        @irdl_op_definition
        class DuplicatedRegionOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_region_op"
            r = region_def()

            assembly_format = "$r $r attr-dict"


def test_attr_dict_directly_before_region_variable():
    """Test that regions require an 'attr-dict' directive."""
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="An `attr-dict' directive without keyword cannot be directly followed by a region variable",
    ):

        @irdl_op_definition
        class RegionAttrDictWrongOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.region_op_missing_keyword"
            region = region_def()

            assembly_format = "attr-dict $region"


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "$region attr-dict",
            "test.region_attr_dict {\n} {a = 2 : i32}",
            '"test.region_attr_dict"() ({}) {a = 2 : i32} : () -> ()',
        ),
        (
            "attr-dict `,` $region",
            'test.region_attr_dict {a = 2 : i32}, {\n  "test.op"() : () -> ()\n}',
            '"test.region_attr_dict"() ({  "test.op"() : () -> ()}) {a = 2 : i32} : () -> ()',
        ),
        (
            "($region^)? attr-dict-with-keyword",
            "test.region_attr_dict attributes {a = 2 : i32}",
            '"test.region_attr_dict"() ({}) {a = 2 : i32} : () -> ()',
        ),
        (
            "($region^)? attr-dict-with-keyword",
            'test.region_attr_dict {\n  "test.op"() : () -> ()\n} attributes {a = 2 : i32}',
            '"test.region_attr_dict"() ({\n  "test.op"() : () -> ()\n}) {a = 2 : i32} : () -> ()',
        ),
    ],
)
def test_regions_with_attr_dict(format: str, program: str, generic_program: str):
    """Test the parsing of regions"""

    @irdl_op_definition
    class RegionsOp(IRDLOperation):
        name = "test.region_attr_dict"
        region = region_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(RegionsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "attr-dict-with-keyword $fst $snd",
            "test.two_regions {\n} {\n}",
            '"test.two_regions"() ({}, {}) : () -> ()',
        ),
        (
            "attr-dict-with-keyword $fst $snd",
            'test.two_regions {\n  "test.op"() : () -> ()\n} {\n  "test.op"() : () -> ()\n}',
            '"test.two_regions"() ({ "test.op"() : () -> ()}, { "test.op"() : () -> ()}) : () -> ()',
        ),
        (
            "attr-dict-with-keyword $fst $snd",
            'test.two_regions attributes {a = 2 : i32} {\n  "test.op"() : () -> ()\n} {\n  "test.op"() : () -> ()\n}',
            '"test.two_regions"() ({ "test.op"() : () -> ()}, { "test.op"() : () -> ()}) {a = 2 : i32} : () -> ()',
        ),
    ],
)
def test_regions(format: str, program: str, generic_program: str):
    """Test the parsing of regions"""

    @irdl_op_definition
    class TwoRegionsOp(IRDLOperation):
        name = "test.two_regions"
        fst = region_def()
        snd = region_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(TwoRegionsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "attr-dict-with-keyword $region",
            "test.variadic_region",
            '"test.variadic_region"() : () -> ()',
        ),
        (
            "attr-dict-with-keyword $region",
            'test.variadic_region {\n  "test.op"() : () -> ()\n}',
            '"test.variadic_region"() ({ "test.op"() : () -> ()}) : () -> ()',
        ),
        (
            "attr-dict-with-keyword $region",
            'test.variadic_region {\n  "test.op"() : () -> ()\n} {\n  "test.op"() : () -> ()\n}',
            '"test.variadic_region"() ({ "test.op"() : () -> ()}, { "test.op"() : () -> ()}) : () -> ()',
        ),
        (
            "attr-dict-with-keyword $region",
            'test.variadic_region {\n  "test.op"() : () -> ()\n} {\n  "test.op"() : () -> ()\n} {\n  "test.op"() : () -> ()\n}',
            '"test.variadic_region"() ({ "test.op"() : () -> ()}, {"test.op"() : () -> ()}, {\n  "test.op"() : () -> ()\n}) : () -> ()',
        ),
    ],
)
def test_variadic_region(format: str, program: str, generic_program: str):
    """Test the parsing of variadic regions"""

    @irdl_op_definition
    class VariadicRegionOp(IRDLOperation):
        name = "test.variadic_region"
        region = var_region_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(VariadicRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "attr-dict-with-keyword $region",
            "test.optional_region",
            '"test.optional_region"() : () -> ()',
        ),
        (
            "attr-dict-with-keyword $region",
            'test.optional_region {\n  "test.op"() : () -> ()\n}',
            '"test.optional_region"() ({ "test.op"() : () -> ()}) : () -> ()',
        ),
    ],
)
def test_optional_region(format: str, program: str, generic_program: str):
    """Test the parsing of optional regions"""

    @irdl_op_definition
    class OptionalRegionOp(IRDLOperation):
        name = "test.optional_region"
        region = opt_region_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(OptionalRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


def test_multiple_optional_regions():
    """Test that a variadic region variable cannot directly follow another variadic region variable."""
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="An optional/variadic region variable cannot be followed by another region variable.",
    ):

        @irdl_op_definition
        class OptionalRegionsOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.optional_regions"
            irdl_options = [AttrSizedRegionSegments()]
            region1 = opt_region_def()
            region2 = opt_region_def()

            assembly_format = "attr-dict-with-keyword $region1 $region2"


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "(`keyword` $opt_region^)? attr-dict",
            "test.optional_region_group",
            '"test.optional_region_group"() : () -> ()',
        ),
        (
            "(`keyword` $opt_region^)? attr-dict",
            'test.optional_region_group keyword {\n  "test.op"() : () -> ()\n}',
            '"test.optional_region_group"() ({ "test.op"() : () -> ()}) : () -> ()',
        ),
    ],
)
def test_optional_groups_regions(format: str, program: str, generic_program: str):
    """Test the parsing of optional regions in an optional group"""

    @irdl_op_definition
    class OptionalRegionOp(IRDLOperation):
        name = "test.optional_region_group"
        opt_region = opt_region_def()

        assembly_format = format

    ctx = Context()
    ctx.load_op(OptionalRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.empty_region_group",
            '"test.empty_region_group"() ({}) : () -> ()',
        ),
        (
            "test.empty_region_group keyword {\n^bb0:\n}",
            '"test.empty_region_group"() ({^bb0:}) : () -> ()',
        ),
        (
            'test.empty_region_group keyword {\n  "test.op"() : () -> ()\n}',
            '"test.empty_region_group"() ({ "test.op"() : () -> ()}) : () -> ()',
        ),
    ],
)
def test_optional_groups_empty_regions(program: str, generic_program: str):
    """Test the parsing of empty regions in an optional group"""

    @irdl_op_definition
    class EmptyRegionOp(IRDLOperation):
        name = "test.empty_region_group"
        maybe_empty = region_def()

        assembly_format = "(`keyword` $maybe_empty^)? attr-dict"

    ctx = Context()
    ctx.load_op(EmptyRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


def test_attr_dict_directly_after_optional_group_with_first_region_variable():
    """Test that regions require an 'attr-dict' directive."""
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="An optional group with a region as a first element cannot be followed by a `attr-dict' directive as it is ambiguous.",
    ):

        @irdl_op_definition
        class RegionAttrDictWrongOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.region_op_ambiguous_optional_group"
            region = region_def()

            assembly_format = "($region^)? attr-dict"


################################################################################
# Successors                                                                   #
################################################################################


def test_missing_successor():
    """Test that successors should be parsed."""
    with pytest.raises(PyRDLOpDefinitionError, match="successor 'successor' not found"):

        @irdl_op_definition
        class NoSuccessorOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.no_successor_op"
            successor = successor_def()

            assembly_format = "attr-dict-with-keyword"


def test_successor_duplicated():
    """Test that successors should not be parsed twice"""
    with pytest.raises(
        PyRDLOpDefinitionError, match="successor 'succ' is already bound"
    ):

        @irdl_op_definition
        class DuplicatedSucessorOp(  # pyright: ignore[reportUnusedClass]
            IRDLOperation
        ):
            name = "test.duplicated_successor_op"
            succ = successor_def()

            assembly_format = "$succ $succ attr-dict"


def test_successors():
    """Test the parsing of successors"""

    program = textwrap.dedent(
        """\
        "test.op"() ({
          "test.op"() [^bb0] : () -> ()
        ^bb0:
          test.two_successors ^bb0 ^bb0
        }) : () -> ()"""
    )

    generic_program = textwrap.dedent(
        """\
        "test.op"() ({
          "test.op"() [^bb0] : () -> ()
        ^bb0:
          "test.two_successors"() [^bb0, ^bb0] : () -> ()
        }) : () -> ()"""
    )

    @irdl_op_definition
    class TwoSuccessorsOp(IRDLOperation):
        name = "test.two_successors"
        fst = successor_def()
        snd = successor_def()

        assembly_format = "$fst $snd attr-dict"

    ctx = Context()
    ctx.load_op(TwoSuccessorsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '"test.op"() ({\n  "test.op"() [^bb0] : () -> ()\n^bb0:\n  test.var_successor\n}) : () -> ()',
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  "test.var_successor"() : () -> ()
                }) : () -> ()"""
            ),
        ),
        (
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  test.var_successor ^bb0
                }) : () -> ()"""
            ),
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  "test.var_successor"() [^bb0] : () -> ()
                }) : () -> ()"""
            ),
        ),
        (
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  test.var_successor ^bb0, ^bb0
                }) : () -> ()"""
            ),
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  "test.var_successor"() [^bb0, ^bb0] : () -> ()
                }) : () -> ()"""
            ),
        ),
    ],
)
def test_variadic_successor(program: str, generic_program: str):
    """Test the parsing of successors"""

    @irdl_op_definition
    class VarSuccessorOp(IRDLOperation):
        name = "test.var_successor"
        succ = var_successor_def()

        assembly_format = "$succ attr-dict"

    ctx = Context()
    ctx.load_op(VarSuccessorOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '"test.op"() ({\n  "test.op"() [^bb0] : () -> ()\n^bb0:\n  test.opt_successor\n}) : () -> ()',
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  "test.opt_successor"() : () -> ()
                }) : () -> ()"""
            ),
        ),
        (
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  test.opt_successor ^bb0
                }) : () -> ()"""
            ),
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^bb0] : () -> ()
                ^bb0:
                  "test.opt_successor"() [^bb0] : () -> ()
                }) : () -> ()"""
            ),
        ),
    ],
)
def test_optional_successor(program: str, generic_program: str):
    """Test the parsing of successors"""

    @irdl_op_definition
    class OptSuccessorOp(IRDLOperation):
        name = "test.opt_successor"
        succ = opt_successor_def()

        assembly_format = "$succ attr-dict"

    ctx = Context()
    ctx.load_op(OptSuccessorOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


################################################################################
# Inference                                                                   #
################################################################################

_T = TypeVar("_T", bound=Attribute, covariant=True, default=Attribute)


@pytest.mark.parametrize(
    "format",
    [
        "$lhs $rhs attr-dict `:` type($lhs)",
        "$lhs $rhs attr-dict `:` type($rhs)",
        "$lhs $rhs attr-dict `:` type($res)",
    ],
)
def test_basic_inference(format: str):
    """Check that we can infer the type of an operand when ConstraintVar are used."""

    @irdl_op_definition
    class TwoOperandsOneResultWithVarOp(IRDLOperation):
        T: ClassVar = VarConstraint("T", AnyAttr())

        name = "test.two_operands_one_result_with_var"
        res = result_def(T)
        lhs = operand_def(T)
        rhs = operand_def(T)

        assembly_format = format

    ctx = Context()
    ctx.load_op(TwoOperandsOneResultWithVarOp)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0, %1 = "test.op"() : () -> (i32, i32)
    %2 = test.two_operands_one_result_with_var %0 %1 : i32
    "test.op"(%2) : (i32) -> ()"""
    )
    check_roundtrip(program, ctx)


def test_eq_attr_inference():
    """Check that operands/results with a fixed type can be inferred."""

    @irdl_attr_definition
    class UnitType(ParametrizedAttribute, TypeAttribute):
        name = "test.unit"

    @irdl_op_definition
    class OneOperandEqTypeOp(IRDLOperation):
        name = "test.one_operand_eq_type"
        index = operand_def(UnitType())
        res = result_def(UnitType())

        assembly_format = "attr-dict $index"

    ctx = Context()
    ctx.load_attr_or_type(UnitType)
    ctx.load_op(OneOperandEqTypeOp)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0 = "test.op"() : () -> !test.unit
    %1 = test.one_operand_eq_type %0
    "test.op"(%1) : (!test.unit) -> ()"""
    )
    check_roundtrip(program, ctx)


def test_all_of_attr_inference():
    """Check that AllOf still allows for inference."""

    @irdl_attr_definition
    class UnitType(ParametrizedAttribute, TypeAttribute):
        name = "test.unit"

    @irdl_op_definition
    class OneOperandEqTypeAllOfNestedOp(IRDLOperation):
        name = "test.one_operand_eq_type_all_of_nested"
        index = operand_def(AllOf((AnyAttr(), EqAttrConstraint(UnitType()))))

        assembly_format = "attr-dict $index"

    ctx = Context()
    ctx.load_attr_or_type(UnitType)
    ctx.load_op(OneOperandEqTypeAllOfNestedOp)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0 = "test.op"() : () -> !test.unit
    test.one_operand_eq_type_all_of_nested %0"""
    )
    check_roundtrip(program, ctx)


def test_nested_inference():
    """Check that Param<T> infers correctly T."""

    @irdl_attr_definition
    class ParamOne(ParametrizedAttribute, TypeAttribute, Generic[_T]):
        name = "test.param_one"

        n: Attribute
        p: _T
        q: Attribute

        @staticmethod
        def constr(
            *,
            n: AttrConstraint | None = None,
            p: AttrConstraint[_T] | None = None,
            q: AttrConstraint | None = None,
        ) -> BaseAttr[ParamOne[_T]] | ParamAttrConstraint[ParamOne[_T]]:
            if n is None and p is None and q is None:
                return BaseAttr[ParamOne[_T]](ParamOne)
            return ParamAttrConstraint[ParamOne[_T]](ParamOne, (n, p, q))

    @irdl_op_definition
    class TwoOperandsNestedVarOp(IRDLOperation):
        T: ClassVar = VarConstraint("T", AnyAttr())

        name = "test.two_operands_one_result_with_var"
        res = result_def(T)
        lhs = operand_def(ParamOne.constr(p=T))
        rhs = operand_def(T)

        assembly_format = "$lhs $rhs attr-dict `:` type($lhs)"

    ctx = Context()
    ctx.load_op(TwoOperandsNestedVarOp)
    ctx.load_attr_or_type(ParamOne)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0, %1 = "test.op"() : () -> (!test.param_one<f16, i32, i1>, i32)
    %2 = test.two_operands_one_result_with_var %0 %1 : !test.param_one<f16, i32, i1>"""
    )
    check_roundtrip(program, ctx)


def test_nested_inference_variable():
    """Check that Param<T> infers correctly T when Param<T> is nested in a variable."""

    @irdl_attr_definition
    class ParamOne(ParametrizedAttribute, TypeAttribute, Generic[_T]):
        name = "test.param_one"

        p: _T

        @staticmethod
        def constr(
            *, p: AttrConstraint[_T] | None = None
        ) -> ParamAttrConstraint[ParamOne[_T]]:
            return ParamAttrConstraint[ParamOne[_T]](ParamOne, (p,))

    @irdl_op_definition
    class ResultTypeIsOperandParamOp(IRDLOperation):
        T: ClassVar = VarConstraint("T", AnyAttr())
        U: ClassVar = VarConstraint("U", ParamOne.constr(p=T))

        name = "test.result_type_is_operand_param"
        res = result_def(T)
        arg = operand_def(U)

        assembly_format = "$arg attr-dict `:` type($arg)"

    ctx = Context()
    ctx.load_op(ResultTypeIsOperandParamOp)
    ctx.load_attr_or_type(ParamOne)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0 = "test.op"() : () -> !test.param_one<i32>
    %1 = test.result_type_is_operand_param %0 : !test.param_one<i32>"""
    )
    check_roundtrip(program, ctx)


def test_non_verifying_inference():
    """
    Check that non-verifying operands/results will
    trigger a ParseError when inference is required.
    """

    @irdl_attr_definition
    class ParamOne(ParametrizedAttribute, TypeAttribute, Generic[_T]):
        name = "test.param_one"
        p: _T

        @staticmethod
        def constr(
            *,
            p: AttrConstraint[_T] | None = None,
        ) -> BaseAttr[ParamOne[_T]] | ParamAttrConstraint[ParamOne[_T]]:
            if p is None:
                return BaseAttr[ParamOne[_T]](ParamOne)
            return ParamAttrConstraint[ParamOne[_T]](ParamOne, (p,))

    @irdl_op_definition
    class OneOperandOneResultNestedOp(IRDLOperation):
        T: ClassVar = VarConstraint("T", AnyAttr())

        name = "test.one_operand_one_result_nested"
        res = result_def(T)
        lhs = operand_def(ParamOne.constr(p=T))

        assembly_format = "$lhs attr-dict `:` type($lhs)"

    ctx = Context()
    ctx.load_op(OneOperandOneResultNestedOp)
    ctx.load_attr_or_type(ParamOne)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0 = "test.op"() : () -> i32
    %1 = test.one_operand_one_result_nested %0 : i32"""
    )
    with pytest.raises(
        VerifyException,
        match="i32 should be of base attribute test.param_one",
    ):
        parser = Parser(ctx, program)
        while (op := parser.parse_optional_operation()) is not None:
            op.verify()


def test_variadic_length_inference():
    @irdl_op_definition
    class RangeVarOp(IRDLOperation):
        name = "test.range_var"
        T: ClassVar = RangeVarConstraint("T", RangeOf(AnyAttr()))
        ins = var_operand_def(T)
        outs = var_result_def(T)

        assembly_format = "$ins attr-dict `:` type($ins)"

    ctx = Context()
    ctx.load_op(RangeVarOp)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %in0, %in1 = "test.op"() : () -> (index, index)
    %out0, %out1 = test.range_var %in0, %in1 : index, index
    """
    )

    parser = Parser(ctx, program)
    test_op = parser.parse_optional_operation()
    assert isinstance(test_op, test.Operation)
    my_op = parser.parse_optional_operation()
    assert isinstance(my_op, RangeVarOp)


def test_int_var_inference():
    @irdl_op_definition
    class IntVarOp(IRDLOperation):
        name = "test.int_var"
        T: ClassVar = IntVarConstraint("T", AnyInt())
        ins = var_operand_def(RangeOf(eq(IndexType())).of_length(T))
        outs = var_result_def(RangeOf(eq(IntegerType(64))).of_length(T))

        assembly_format = "$ins attr-dict"

    ctx = Context()
    ctx.load_op(IntVarOp)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %in0, %in1 = "test.op"() : () -> (index, index)
    %out0, %out1 = test.int_var %in0, %in1
    """
    )

    parser = Parser(ctx, program)
    test_op = parser.parse_optional_operation()
    assert isinstance(test_op, test.Operation)
    my_op = parser.parse_optional_operation()
    assert isinstance(my_op, IntVarOp)
    assert my_op.result_types == (IntegerType(64), IntegerType(64))


################################################################################
# Declarative Format Verification                                              #
################################################################################


@pytest.mark.parametrize(
    "variadic_def, format",
    [
        (var_operand_def, "$variadic `,` type($variadic) attr-dict"),
        (var_operand_def, "$variadic `:` type($variadic) `,` attr-dict"),
        (var_result_def, "type($variadic) `,` attr-dict"),
    ],
)
def test_variadic_comma_safeguard(
    variadic_def: Callable[[], VarOperand | VarOpResult], format: str
):
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="A variadic directive cannot be followed by a comma literal.",
    ):

        @irdl_op_definition
        class CommaSafeguardOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.comma_safeguard"

            variadic = variadic_def()
            assembly_format = format


@pytest.mark.parametrize(
    "variadic_def_one, variadic_def_two, format",
    [
        (
            var_operand_def,
            var_operand_def,
            "$variadic_one `and` $variadic_two `:` type($variadic_one) type($variadic_two) attr-dict",
        ),
        (
            var_result_def,
            var_result_def,
            "type($variadic_one) type($variadic_two) attr-dict",
        ),
        (
            opt_operand_def,
            opt_operand_def,
            "$variadic_one `and` $variadic_two `:` type($variadic_one) type($variadic_two) attr-dict",
        ),
        (
            opt_result_def,
            opt_result_def,
            "type($variadic_one) type($variadic_two) attr-dict",
        ),
    ],
)
def test_chained_variadic_types_safeguard(
    variadic_def_one: Callable[[], VarOperand | VarOpResult],
    variadic_def_two: Callable[[], VarOperand | VarOpResult],
    format: str,
):
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="An optional/variadic type directive cannot be followed by another type directive.",
    ):

        @irdl_op_definition
        class VarTypeGuardOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.variadic_type_safeguard"

            variadic_one = variadic_def_one()
            variadic_two = variadic_def_two()
            assembly_format = format

            irdl_options = [AttrSizedOperandSegments(), AttrSizedResultSegments()]


@pytest.mark.parametrize("variadic_def_one", [var_operand_def, opt_operand_def])
@pytest.mark.parametrize("variadic_def_two", [var_operand_def, opt_operand_def])
def test_chained_variadic_operands_safeguard(
    variadic_def_one: Callable[[], VarOperand | VarOpResult],
    variadic_def_two: Callable[[], VarOperand | VarOpResult],
):
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="An optional/variadic operand variable cannot be followed by another operand variable.",
    ):

        @irdl_op_definition
        class VarOpGuardOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.variadic_operand_safeguard"

            variadic_one = variadic_def_one()
            variadic_two = variadic_def_two()
            assembly_format = "$variadic_one $variadic_two `:` type($variadic_one) `<` type($variadic_two) `>` attr-dict"

            irdl_options = [AttrSizedOperandSegments()]


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '%0 = "test.op"() : () -> i32\ntest.optional_group(%0 : i32)',
            '%0 = "test.op"() : () -> i32\n"test.optional_group"(%0) : (i32) -> ()',
        ),
        (
            "test.optional_group",
            '"test.optional_group"() : () -> ()',
        ),
    ],
)
def test_optional_group_optional_operand_anchor(
    program: str,
    generic_program: str,
):
    @irdl_op_definition
    class OptionalGroupOp(IRDLOperation):
        name = "test.optional_group"

        args = opt_operand_def()

        assembly_format = "(`(` $args^ `:` type($args) `)`)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalGroupOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '%0 = "test.op"() : () -> i32\ntest.optional_else_group %0',
            '%0 = "test.op"() : () -> i32\n"test.optional_else_group"(%0) : (i32) -> ()',
        ),
        (
            "test.optional_else_group 1",
            '"test.optional_else_group"() <{a = 1 : i32}> : () -> ()',
        ),
    ],
)
def test_optional_else_group(
    program: str,
    generic_program: str,
):
    @irdl_op_definition
    class OptionalElseGroupOp(IRDLOperation):
        name = "test.optional_else_group"

        v = opt_operand_def(i32)
        a = opt_prop_def(IntegerAttr[I32])

        assembly_format = """($v^):($a)? attr-dict"""

    ctx = Context()
    ctx.load_op(OptionalElseGroupOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


def test_impossible_optional_else_group():
    error = "property 'val' is already bound"
    with pytest.raises(
        PyRDLOpDefinitionError,
        match=error,
    ):

        @irdl_op_definition
        class OptionalImpossibleElseGroup(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.impossible_optional_else_group"

            val = opt_prop_def(IntegerAttr[I32])

            assembly_format = """($val^):($val)? attr-dict"""


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '%0 = "test.op"() : () -> i32\ntest.optional_optional_group with "prop" %0',
            '%0 = "test.op"() : () -> i32\n"test.optional_optional_group"(%0) <{prop="prop"}> : (i32) -> ()',
        ),
        (
            'test.optional_optional_group with "prop"',
            '"test.optional_optional_group"() <{prop="prop"}>: () -> ()',
        ),
        (
            "test.optional_optional_group",
            '"test.optional_optional_group"() : () -> ()',
        ),
    ],
)
def test_optional_optional_group_optional_operand_anchor(
    program: str,
    generic_program: str,
):
    @irdl_op_definition
    class OptionalOptionalGroupOp(IRDLOperation):
        name = "test.optional_optional_group"

        prop = opt_prop_def(StringAttr)
        arg = opt_operand_def(I32)

        assembly_format = "(`with` $prop^ ($arg^)?)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalOptionalGroupOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.optional_group %0, %1 : i32, i64",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.optional_group"(%0, %1) : (i32, i64) -> ()',
        ),
        (
            '%0 = "test.op"() : () -> i32\ntest.optional_group %0 : i32',
            '%0 = "test.op"() : () -> i32\n"test.optional_group"(%0) : (i32) -> ()',
        ),
        (
            "test.optional_group",
            '"test.optional_group"() : () -> ()',
        ),
    ],
)
def test_optional_group_variadic_operand_anchor(
    program: str,
    generic_program: str,
):
    @irdl_op_definition
    class OptionalGroupOp(IRDLOperation):
        name = "test.optional_group"

        args = var_operand_def()

        assembly_format = "($args^ `:` type($args))? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalGroupOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "%0 = test.optional_group(i32)",
            '%0 = "test.optional_group"() : () -> (i32)',
        ),
        (
            "test.optional_group",
            '"test.optional_group"() : () -> ()',
        ),
    ],
)
def test_optional_group_optional_result_anchor(
    program: str,
    generic_program: str,
):
    @irdl_op_definition
    class OptionalGroupOp(IRDLOperation):
        name = "test.optional_group"

        res = opt_result_def()

        assembly_format = "(`(` type($res)^ `)`)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalGroupOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "%0, %1 = test.optional_group(i32, i64)",
            '%0, %1 = "test.optional_group"() : () -> (i32, i64)',
        ),
        (
            "%0 = test.optional_group(i32)",
            '%0 = "test.optional_group"() : () -> (i32)',
        ),
        (
            "test.optional_group",
            '"test.optional_group"() : () -> ()',
        ),
    ],
)
def test_optional_group_variadic_result_anchor(
    program: str,
    generic_program: str,
):
    @irdl_op_definition
    class OptionalGroupOp(IRDLOperation):
        name = "test.optional_group"

        res = var_result_def()

        assembly_format = "(`(` type($res)^ `)`)? attr-dict"

    ctx = Context()
    ctx.load_op(OptionalGroupOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, error",
    [
        ("()?", "An optional group must have a non-whitespace directive"),
        ("(`keyword`)?", "Every optional group must have an anchor."),
        (
            "($args^ type($rets)^)?",
            "An optional group can only have one anchor.",
        ),
        (
            "(`keyword`^)?",
            "An optional group's anchor must be an anchorable directive.",
        ),
        (
            "($mandatory_arg^)?",
            "First element of an optional group must be optionally parsable.",
        ),
        (
            "($optional_unit_arg^)?",
            "First element of an optional group must be optionally parsable.",
        ),
    ],
)
def test_optional_group_checkers(format: str, error: str):
    with pytest.raises(
        PyRDLOpDefinitionError,
        match=error,
    ):

        @irdl_op_definition
        class WrongOptionalGroupOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.wrong_optional_group"

            args = var_operand_def()
            rets = var_result_def()
            mandatory_arg = operand_def()
            optional_unit_arg = opt_prop_def(UnitAttr())

            assembly_format = format


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '%0 = "test.op"() : () -> !test.type<"index">\ntest.mixed %0()',
            '%0 = "test.op"() : () -> !test.type<"index">\n'
            '"test.mixed"(%0) : (!test.type<"index">) -> ()',
        ),
        (
            '%0 = "test.op"() : () -> !test.type<"index">\ntest.mixed %0(%0)',
            '%0 = "test.op"() : () -> !test.type<"index">\n'
            '"test.mixed"(%0, %0) : (!test.type<"index">, !test.type<"index">) -> ()',
        ),
        (
            '%0 = "test.op"() : () -> !test.type<"index">\ntest.mixed %0(%0, %0)',
            '%0 = "test.op"() : () -> !test.type<"index">\n'
            '"test.mixed"(%0, %0, %0) : (!test.type<"index">, !test.type<"index">, !test.type<"index">) -> ()',
        ),
    ],
)
def test_variadic_and_single_mixed(program: str, generic_program: str):
    @irdl_op_definition
    class MixedOp(IRDLOperation):
        name = "test.mixed"
        var = var_operand_def(TestType("index"))
        sin = operand_def(TestType("index"))

        assembly_format = "$sin `(` $var `)` attr-dict"

    ctx = Context()
    ctx.load_op(MixedOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@irdl_op_definition
class DefaultOp(IRDLOperation):
    name = "test.default"

    prop = prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    opt_prop = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(True))

    attr = attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    opt_attr = opt_attr_def(BoolAttr, default_value=BoolAttr.from_bool(True))

    assembly_format = "(`prop` $prop^)? (`opt_prop` $opt_prop^)? (`attr` $attr^)? (`opt_attr` $opt_attr^)? attr-dict"


@pytest.mark.parametrize(
    "program, output, generic",
    [
        (
            "test.default",
            "test.default",
            '"test.default"() <{prop = false}> {attr = false} : () -> ()',
        ),
        (
            "test.default prop false opt_prop true",
            "test.default",
            '"test.default"() <{prop = false, opt_prop = true}> {attr = false} : () -> ()',
        ),
        (
            '"test.default"() <{prop = false, opt_prop = true}> {attr = false} : () -> ()',
            "test.default",
            '"test.default"() <{prop = false, opt_prop = true}> {attr = false} : () -> ()',
        ),
        (
            '"test.default"() <{prop = false}> {attr = false} : () -> ()',
            "test.default",
            '"test.default"() <{prop = false}> {attr = false} : () -> ()',
        ),
        (
            "test.default prop true opt_prop false",
            "test.default prop true opt_prop false",
            '"test.default"() <{prop = true, opt_prop = false}> {attr = false} : () -> ()',
        ),
        (
            "test.default attr false opt_attr true",
            "test.default",
            '"test.default"() <{prop = false}> {attr = false, opt_attr = true} : () -> ()',
        ),
        (
            '"test.default"() <{prop = false}> {attr = false, opt_attr = true} : () -> ()',
            "test.default",
            '"test.default"() <{prop = false}> {attr = false, opt_attr = true} : () -> ()',
        ),
        (
            "test.default attr true opt_attr false",
            "test.default attr true opt_attr false",
            '"test.default"() <{prop = false}> {attr = true, opt_attr = false} : () -> ()',
        ),
        (
            '"test.default"() : () -> ()',
            "test.default",
            '"test.default"() <{prop = false}> {attr = false} : () -> ()',
        ),
    ],
)
def test_default_properties(program: str, output: str, generic: str):
    ctx = Context()
    ctx.load_op(DefaultOp)

    parsed = Parser(ctx, program).parse_operation()
    assert isinstance(parsed, DefaultOp)

    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_op(parsed)

    assert output == stream.getvalue()

    stream = StringIO()
    printer = Printer(stream=stream, print_generic_format=True)
    printer.print_op(parsed)

    assert generic == stream.getvalue()


@irdl_op_definition
class RenamedPropOp(IRDLOperation):
    name = "test.renamed"

    prop1 = prop_def(
        BoolAttr, default_value=BoolAttr.from_bool(False), prop_name="test_prop1"
    )
    prop2 = opt_prop_def(BoolAttr, prop_name="test_prop2")

    assembly_format = "(`prop1` $test_prop1^)? (`prop2` $test_prop2^)? attr-dict"


@pytest.mark.parametrize(
    "program, output, generic",
    [
        (
            "test.renamed",
            "test.renamed",
            '"test.renamed"() <{test_prop1 = false}> : () -> ()',
        ),
        (
            "test.renamed prop1 false prop2 false",
            "test.renamed prop2 false",
            '"test.renamed"() <{test_prop1 = false, test_prop2 = false}> : () -> ()',
        ),
        (
            "test.renamed prop1 true prop2 true",
            "test.renamed prop1 true prop2 true",
            '"test.renamed"() <{test_prop1 = true, test_prop2 = true}> : () -> ()',
        ),
    ],
)
def test_renamed_optional_prop(program: str, output: str, generic: str):
    ctx = Context()
    ctx.load_op(RenamedPropOp)

    parsed = Parser(ctx, program).parse_operation()
    assert isinstance(parsed, RenamedPropOp)

    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_op(parsed)

    assert output == stream.getvalue()

    stream = StringIO()
    printer = Printer(stream=stream, print_generic_format=True)
    printer.print_op(parsed)

    assert generic == stream.getvalue()


@pytest.mark.parametrize(
    "program, generic",
    [
        (
            "test.opt_constant : ()",
            '"test.opt_constant"() : () -> ()',
        ),
        (
            "%0 = test.opt_constant value 1 : i32 : (i32)",
            '%0 = "test.opt_constant"() <{value = 1 : i32}> : () -> (i32)',
        ),
    ],
)
def test_optional_property_with_extractor(program: str, generic: str):
    with pytest.deprecated_call():

        @irdl_op_definition
        class OptConstantOp(IRDLOperation):
            name = "test.opt_constant"
            T: ClassVar = VarConstraint("T", AnyAttr())

            value = opt_prop_def(TypedAttributeConstraint(IntegerAttr.constr(), T))  # pyright: ignore[reportDeprecated]

            res = opt_result_def(T)

            assembly_format = "(`value` $value^)? attr-dict `:` `(` type($res) `)`"

    ctx = Context()
    ctx.load_op(OptConstantOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic, ctx)


@pytest.mark.parametrize(
    "program, generic",
    [
        (
            "%0 = test.default_constant",
            '%0 = "test.default_constant"() <{value = true}> : () -> (i1)',
        ),
        (
            "%0 = test.default_constant value 2 : i32",
            '%0 = "test.default_constant"() <{value = 2 : i32}> : () -> (i32)',
        ),
    ],
)
def test_default_property_with_extractor(program: str, generic: str):
    with pytest.deprecated_call():

        @irdl_op_definition
        class DefaultConstantOp(IRDLOperation):
            name = "test.default_constant"
            T: ClassVar = VarConstraint("T", AnyAttr())

            value = prop_def(
                TypedAttributeConstraint(IntegerAttr.constr(), T),  # pyright: ignore[reportDeprecated]
                default_value=BoolAttr.from_bool(True),
            )

            res = result_def(T)

            assembly_format = "(`value` $value^)? attr-dict"

    ctx = Context()
    ctx.load_op(DefaultConstantOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic, ctx)


@pytest.mark.parametrize(
    "program, generic",
    [
        (
            "test.default_attr_dict",
            '"test.default_attr_dict"() <{prop = false}> {attr = false} : () -> ()',
        ),
        (
            "test.default_attr_dict {attr = true, prop = true}",
            '"test.default_attr_dict"() <{prop = true}> {attr = true} : () -> ()',
        ),
    ],
)
def test_default_property_in_attr_dict(program: str, generic: str):
    @irdl_op_definition
    class DefaultAttrDictOp(IRDLOperation):
        name = "test.default_attr_dict"

        prop = prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

        attr = attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))

        irdl_options = [ParsePropInAttrDict()]

        assembly_format = "attr-dict"

    ctx = Context()
    ctx.load_op(DefaultAttrDictOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic, ctx)


@pytest.mark.parametrize(
    "program, generic",
    [
        (
            "test.default_attr_dict",
            '"test.default_attr_dict"() {attr = false} : () -> ()',
        ),
        (
            "test.default_attr_dict {attr = true}",
            '"test.default_attr_dict"() {attr = true} : () -> ()',
        ),
    ],
)
def test_default_attr_in_attr_dict(program: str, generic: str):
    @irdl_op_definition
    class DefaultAttrDictOp(IRDLOperation):
        name = "test.default_attr_dict"

        attr = attr_def(BoolAttr, default_value=BoolAttr.from_bool(False))

        assembly_format = "attr-dict"

    ctx = Context()
    ctx.load_op(DefaultAttrDictOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic, ctx)


################################################################################
#                            Variable extraction                               #
################################################################################


@irdl_op_definition
class AllOfExtractorOp(IRDLOperation):
    name = "test.all_of_extractor"

    T: ClassVar = VarConstraint("T", AnyAttr())
    lhs = operand_def(T & MemRefType.constr(T))
    rhs = operand_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"


def test_all_of_extraction_fails():
    ctx = Context()
    ctx.load_op(AllOfExtractorOp)
    ctx.load_dialect(Test)
    parser = Parser(
        ctx,
        '%0 = "test.op"() : () -> memref<10xindex>\ntest.all_of_extractor %0, %0 : memref<10xindex>',
    )
    parser.parse_operation()
    with pytest.raises(
        VerifyException,
        match="attribute memref<10xindex> expected from variable 'T', but got index",
    ):
        parser.parse_operation()


@irdl_attr_definition
class DoubleParamAttr(ParametrizedAttribute, TypeAttribute):
    """An attribute with two unbounded attribute parameters."""

    name = "test.param"

    param1: Attribute
    param2: Attribute


@irdl_op_definition
class ParamExtractorOp(IRDLOperation):
    name = "test.param_extractor"

    T: ClassVar = VarConstraint("T", AnyAttr())
    lhs = operand_def(ParamAttrConstraint(DoubleParamAttr, (T, T)))
    rhs = operand_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"


def test_param_extraction_fails():
    ctx = Context()
    ctx.load_attr_or_type(DoubleParamAttr)
    ctx.load_op(ParamExtractorOp)
    ctx.load_dialect(Test)
    parser = Parser(
        ctx,
        '%0 = "test.op"() : () -> !test.param<i32,i64>\ntest.param_extractor %0, %0 : !test.param<i32,i64>',
    )
    parser.parse_operation()
    with pytest.raises(
        VerifyException,
        match="attribute i32 expected from variable 'T', but got i64",
    ):
        parser.parse_operation()


@irdl_op_definition
class MultipleOperandExtractorOp(IRDLOperation):
    name = "test.multiple_operand_extractor"

    T: ClassVar = VarConstraint("T", AnyAttr())
    lhs = operand_def(T)
    rhs = operand_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)"


def test_multiple_operand_extraction_fails():
    ctx = Context()
    ctx.load_op(MultipleOperandExtractorOp)
    ctx.load_dialect(Test)
    parser = Parser(
        ctx,
        '%0, %1 = "test.op"() : () -> (index, i32)\ntest.multiple_operand_extractor %0, %1 : index, i32',
    )
    parser.parse_operation()
    with pytest.raises(
        VerifyException,
        match="attribute index expected from variable 'T', but got i32",
    ):
        parser.parse_operation()


################################################################################
#                                  IntAttr                                     #
################################################################################


@irdl_op_definition
class IntAttrExtractOp(IRDLOperation):
    name = "test.int_attr_extract"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    prop = prop_def(IntegerAttr.constr(value=_I, type=eq(IndexType())))

    outs = var_result_def(RangeOf(eq(IndexType())).of_length(_I))

    assembly_format = "$prop attr-dict"


@pytest.mark.parametrize(
    "program",
    ["%0 = test.int_attr_extract 1", "%0, %1 = test.int_attr_extract 2"],
)
def test_int_attr_extraction(program: str):
    ctx = Context()
    ctx.load_op(IntAttrExtractOp)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "%0 = test.int_attr_extract 2",
            "Operation has 2 results, but was given 1 to bind",
        ),
        (
            "%0, %1 = test.int_attr_extract 1",
            "Operation has 1 results, but was given 2 to bind",
        ),
    ],
)
def test_int_attr_extraction_errors(program: str, error: str):
    ctx = Context()
    ctx.load_op(IntAttrExtractOp)
    parser = Parser(ctx, program)
    with pytest.raises(ParseError, match=error):
        parser.parse_optional_operation()


@irdl_op_definition
class IntAttrVerifyOp(IRDLOperation):
    name = "test.int_attr_verify"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    prop = prop_def(IntegerAttr.constr(value=_I, type=eq(IndexType())))

    prop2 = opt_prop_def(IntegerAttr.constr(value=_I, type=eq(IndexType())))

    ins = var_operand_def(RangeOf(eq(IndexType())).of_length(_I))

    assembly_format = "$prop (`and` $prop2^)? `,` $ins attr-dict"


@pytest.mark.parametrize(
    "program",
    [
        "test.int_attr_verify 1, %0",
        "test.int_attr_verify 2, %0, %1",
        "test.int_attr_verify 1 and 1, %0",
        "test.int_attr_verify 2 and 2, %0, %1",
    ],
)
def test_int_attr_verify(program: str):
    ctx = Context()
    ctx.load_op(IntAttrVerifyOp)

    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "test.int_attr_verify 1, %0, %1",
            "integer 2 expected from int variable 'I', but got 1",
        ),
        (
            "test.int_attr_verify 1 and 2, %0",
            "integer 1 expected from int variable 'I', but got 2",
        ),
        (
            "test.int_attr_verify 2, %0",
            "integer 1 expected from int variable 'I'",
        ),
        (
            "test.int_attr_verify 2 and 1, %0, %1",
            "integer 2 expected from int variable 'I', but got 1",
        ),
    ],
)
def test_int_attr_verify_errors(program: str, error: str):
    ctx = Context()
    ctx.load_op(IntAttrVerifyOp)

    parser = Parser(ctx, program)
    with pytest.raises(VerifyException, match=error):
        op = parser.parse_operation()
        op.verify()


@irdl_attr_definition
class MyAttr(ParametrizedAttribute):
    name = "test.my_attr"

    param: StringAttr


@irdl_op_definition
class NonQualifiedAttrOp(IRDLOperation):
    name = "test.non_qualified_attr"

    attr = prop_def(MyAttr)

    assembly_format = "$attr attr-dict"


def test_non_qualified_attr():
    ctx = Context()
    ctx.load_op(NonQualifiedAttrOp)
    ctx.load_attr_or_type(MyAttr)
    ctx.load_dialect(Test)
    parser = Parser(ctx, 'test.non_qualified_attr <"test">')
    op = parser.parse_operation()
    assert isinstance(op, NonQualifiedAttrOp)
    assert op.attr == MyAttr(StringAttr("test"))


@irdl_op_definition
class QualifiedAttrOp(IRDLOperation):
    name = "test.qualified_attr"

    attr = prop_def(MyAttr)

    assembly_format = "qualified($attr) attr-dict"


def test_qualified_attr():
    ctx = Context()
    ctx.load_op(QualifiedAttrOp)
    ctx.load_attr_or_type(MyAttr)
    ctx.load_dialect(Test)
    parser = Parser(ctx, 'test.qualified_attr #test.my_attr<"test">')
    op = parser.parse_operation()
    assert isinstance(op, QualifiedAttrOp)
    assert op.attr == MyAttr(StringAttr("test"))


################################################################################
#                             Custom directives                                #
################################################################################


@irdl_custom_directive
class Hello(CustomDirective):
    def parse(self, parser: Parser, state: ParsingState) -> bool:
        parser.parse_keyword("hello")
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        printer.print_string("hello")


@irdl_op_definition
class CustomDirectiveOp(IRDLOperation):
    name = "test.custom"

    assembly_format = "custom<Hello>() attr-dict"

    custom_directives = (Hello,)


@pytest.mark.parametrize(
    "program",
    [
        "test.custom hello",
        "test.custom hello {attr = 1 : i32}",
    ],
)
def test_custom_directive(program: str):
    ctx = Context()
    ctx.load_op(CustomDirectiveOp)
    ctx.load_dialect(Test)
    check_roundtrip(program, ctx)


@irdl_custom_directive
class Bars(CustomDirective):
    """We print the operands with bars between, because why not."""

    var: VariadicOperandVariable

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        first = parser.parse_optional_unresolved_operand()
        if first is None:
            operands = []
        else:
            operands = [first]
            while parser.parse_optional_punctuation("|"):
                operands.append(parser.parse_unresolved_operand())
        self.var.set(state, operands)
        return bool(operands)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.var.get(op)
        if not operands:
            return
        state.print_whitespace(printer)
        printer.print_list(operands, printer.print_ssa_value, delimiter=" | ")


@irdl_op_definition
class CustomDirectiveWithParamOp(IRDLOperation):
    name = "test.custom_param"

    ops = var_operand_def()

    assembly_format = "custom<Bars>($ops) (`:` type($ops)^)? attr-dict"

    custom_directives = (Bars,)


@pytest.mark.parametrize(
    "program",
    [
        "test.custom_param",
        "test.custom_param %0 : i32",
        "test.custom_param %0 | %1 : i32, i32",
        "test.custom_param %0 | %1 | %2 : i32, i32, i32",
    ],
)
def test_custom_directive_param(program: str):
    ctx = Context()
    ctx.load_op(CustomDirectiveWithParamOp)
    ctx.load_dialect(Test)
    check_roundtrip(program, ctx)


def test_non_upper_classvar():
    with pytest.raises(
        PyRDLError, match='Invalid ClassVar name "bad", must be uppercase.'
    ):

        @irdl_custom_directive
        class BadClassVar(CustomDirective):  # pyright: ignore[reportUnusedClass]
            bad: ClassVar

            def parse(self, parser: Parser, state: ParsingState) -> bool:
                raise NotImplementedError()

            def print(
                self, printer: Printer, state: PrintingState, op: IRDLOperation
            ) -> None:
                raise NotImplementedError()


def test_bad_parameter():
    with pytest.raises(
        PyRDLError,
        match="Custom directive BadParam has parameter int_param which is not a format directive.",
    ):

        @irdl_custom_directive
        class BadParam(CustomDirective):  # pyright: ignore[reportUnusedClass]
            int_param: int

            def parse(self, parser: Parser, state: ParsingState) -> bool:
                raise NotImplementedError()

            def print(
                self, printer: Printer, state: PrintingState, op: IRDLOperation
            ) -> None:
                raise NotImplementedError()


@irdl_custom_directive
class EmptyDirectiveWithParams(CustomDirective):
    op: OperandVariable
    op_type: TypeDirective
    res_type: TypeDirective
    attr: AttributeVariable
    prop: AttributeVariable
    region: RegionVariable
    successor: SuccessorVariable
    operands: OperandsDirective
    operand_types: TypeDirective
    results: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        pass


@irdl_op_definition
class RefDirectivesOp(IRDLOperation):
    name = "test.ref_directives"

    op = operand_def()
    res = result_def()
    attr = attr_def()
    prop = prop_def()
    region = region_def()
    successor = successor_def()

    assembly_format = (
        "$op type($op) type($res) $attr $prop $region $successor "
        "custom<EmptyDirectiveWithParams>("
        "ref($op),"
        "ref(type($op)),"
        "ref(type($res)),"
        "ref($attr),"
        "ref($prop),"
        "ref($region),"
        "ref($successor),"
        "ref(operands),"
        "ref(type(operands)),"
        "ref(type(results))"
        ") attr-dict"
    )

    custom_directives = (EmptyDirectiveWithParams,)


def test_ref_directives():
    ctx = Context()
    ctx.load_op(RefDirectivesOp)
    ctx.load_dialect(Test)
    check_roundtrip("%0 = test.ref_directives %1 i1 i2 i3 i4 {\n} ^bb0", ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.dyn_index_list",
            '"test.dyn_index_list"() <{static_indices = array<i64>}> : () -> ()',
        ),
        (
            '%0 = "test.op"() : () -> i64\ntest.dyn_index_list keyword [%0]',
            '%0 = "test.op"() : () -> i64\n'
            '"test.dyn_index_list"(%0) <{static_indices = array<i64: -9223372036854775808>}> : (i64) -> ()',
        ),
        (
            "test.dyn_index_list keyword [3, 5, 7, 9]",
            '"test.dyn_index_list"() <{static_indices = array<i64: 3, 5, 7 ,9>}> : () -> ()',
        ),
        (
            '%0 = "test.op"() : () -> i64\ntest.dyn_index_list keyword [%0, 5, %0, 9]',
            '%0 = "test.op"() : () -> i64\n'
            '"test.dyn_index_list"(%0, %0) <{static_indices = array<i64: -9223372036854775808, 5, -9223372036854775808, 9>}> : (i64, i64) -> ()',
        ),
    ],
)
def test_optional_anchor_dynamic_index_list(program: str, generic_program: str):
    @irdl_op_definition
    class DynIndexListOp(IRDLOperation):
        name = "test.dyn_index_list"

        dynamic_indices = var_operand_def(I64)
        static_indices = prop_def(DenseArrayBase[I64])

        assembly_format = "(`keyword` custom<DynamicIndexList>($dynamic_indices, $static_indices)^)? attr-dict"

        custom_directives = (DynamicIndexList,)

    ctx = Context()
    ctx.load_op(DynIndexListOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)
