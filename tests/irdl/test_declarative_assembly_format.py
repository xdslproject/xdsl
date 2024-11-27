from __future__ import annotations

import textwrap
from collections.abc import Callable
from io import StringIO
from typing import Annotated, ClassVar, Generic, TypeVar

import pytest

from xdsl.context import MLContext
from xdsl.dialects import test
from xdsl.dialects.builtin import (
    I32,
    BoolAttr,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    UnitAttr,
)
from xdsl.dialects.test import Test, TestType
from xdsl.ir import (
    Attribute,
    Operation,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    BaseAttr,
    EqAttrConstraint,
    GenericAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    ParameterDef,
    ParsePropInAttrDict,
    RangeOf,
    RangeVarConstraint,
    VarConstraint,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
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
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, PyRDLOpDefinitionError, VerifyException

################################################################################
# Utils for this test file                                                     #
################################################################################


def check_roundtrip(program: str, ctx: MLContext):
    """Check that the given program roundtrips exactly (including whitespaces)."""
    parser = Parser(ctx, program)
    ops: list[Operation] = []
    while (op := parser.parse_optional_operation()) is not None:
        ops.append(op)

    res_io = StringIO()
    printer = Printer(stream=res_io)
    for op in ops[:-1]:
        printer.print_op(op)
        printer.print("\n")
    printer.print_op(ops[-1])

    assert program == res_io.getvalue()


def check_equivalence(program1: str, program2: str, ctx: MLContext):
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
            'test.attr_dict {"a" = 2 : i32}',
            '"test.attr_dict"() {"a" = 2 : i32} : () -> ()',
        ),
        (
            'test.attr_dict_with_keyword attributes {"a" = 2 : i32}',
            '"test.attr_dict_with_keyword"() {"a" = 2 : i32} : () -> ()',
        ),
    ],
)
def test_attr_dict(program: str, generic_program: str):
    """Test the 'attr-dict' and 'attr-dict-with-keyword' directives."""
    ctx = MLContext()
    ctx.load_op(AttrDictOp)
    ctx.load_op(AttrDictWithKeywordOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ('test.prop {"prop" = true}', '"test.prop"() <{"prop" = true}> : () -> ()'),
        (
            'test.prop {"a" = 2 : i32, "prop" = true}',
            '"test.prop"() <{"prop" = true}> {"a" = 2 : i32} : () -> ()',
        ),
    ],
)
def test_attr_dict_prop_fallack(program: str, generic_program: str):
    @irdl_op_definition
    class PropOp(IRDLOperation):
        name = "test.prop"
        prop = opt_prop_def(Attribute)
        irdl_options = [ParsePropInAttrDict()]
        assembly_format = "attr-dict"

    ctx = MLContext()
    ctx.load_op(PropOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


################################################################################
# Attribute variables                                                          #
################################################################################


@irdl_op_definition
class OpWithAttrOp(IRDLOperation):
    name = "test.one_attr"

    attr = attr_def(Attribute)
    assembly_format = "$attr attr-dict"


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_attr i32", '"test.one_attr"() {"attr" = i32} : () -> ()'),
        (
            'test.one_attr i32 {"attr2" = i64}',
            '"test.one_attr"() {"attr" = i32, "attr2" = i64} : () -> ()',
        ),
    ],
)
def test_standard_attr_directive(program: str, generic_program: str):
    ctx = MLContext()
    ctx.load_op(OpWithAttrOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


def test_attr_variable_shadowed():
    ctx = MLContext()
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
        ("test.one_attr i32", '"test.one_attr"() {"irdl" = i32} : () -> ()'),
        (
            'test.one_attr i32 {"attr2" = i64}',
            '"test.one_attr"() {"irdl" = i32, "attr2" = i64} : () -> ()',
        ),
    ],
)
def test_attr_name(program: str, generic_program: str):
    @irdl_op_definition
    class RenamedAttrOp(IRDLOperation):
        name = "test.one_attr"

        python = attr_def(Attribute, attr_name="irdl")
        assembly_format = "$irdl attr-dict"

    ctx = MLContext()
    ctx.load_op(RenamedAttrOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.one_attr <5 : i64>",
            '"test.one_attr"() {"attr" = #test.param<5 : i64>} : () -> ()',
        ),
        (
            'test.one_attr <"hello">',
            '"test.one_attr"() {"attr" = #test.param<"hello">} : () -> ()',
        ),
        (
            'test.one_attr <#test.param<"nested">>',
            '"test.one_attr"() {"attr" = #test.param<#test.param<"nested">>} : () -> ()',
        ),
    ],
)
def test_unqualified_attr(program: str, generic_program: str):
    @irdl_attr_definition
    class ParamOne(ParametrizedAttribute):
        name = "test.param"
        p: ParameterDef[Attribute]

    @irdl_op_definition
    class UnqualifiedAttrOp(IRDLOperation):
        name = "test.one_attr"

        attr = attr_def(ParamOne)
        assembly_format = "$attr attr-dict"

    ctx = MLContext()
    ctx.load_attr(ParamOne)
    ctx.load_op(UnqualifiedAttrOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


def test_missing_property_error():
    class MissingPropOp(IRDLOperation):
        name = "test.missing_prop"

        prop1 = prop_def(Attribute)
        prop2 = prop_def(Attribute)
        assembly_format = "$prop1 attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="prop2 properties are missing",
    ):
        irdl_op_definition(MissingPropOp)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_prop i32", '"test.one_prop"() <{"prop" = i32}> : () -> ()'),
        (
            'test.one_prop i32 {"attr2" = i64}',
            '"test.one_prop"() <{"prop" = i32}> {"attr2" = i64} : () -> ()',
        ),
    ],
)
def test_standard_prop_directive(program: str, generic_program: str):
    @irdl_op_definition
    class PropOp(IRDLOperation):
        name = "test.one_prop"

        prop = prop_def(Attribute)
        assembly_format = "$prop attr-dict"

    ctx = MLContext()
    ctx.load_op(PropOp)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        ("test.one_prop i32", '"test.one_prop"() <{"irdl" = i32}> : () -> ()'),
        (
            'test.one_prop i32 {"attr2" = i64}',
            '"test.one_prop"() <{"irdl" = i32}> {"attr2" = i64} : () -> ()',
        ),
    ],
)
def test_prop_name(program: str, generic_program: str):
    @irdl_op_definition
    class RenamedPropOp(IRDLOperation):
        name = "test.one_prop"

        python = prop_def(Attribute, prop_name="irdl")
        assembly_format = "$irdl attr-dict"

    ctx = MLContext()
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
            '"test.optional_property"() <{"prop" = i32}> : () -> ()',
        ),
    ],
)
def test_optional_property(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalPropertyOp(IRDLOperation):
        name = "test.optional_property"
        prop = opt_prop_def(Attribute)

        assembly_format = "(`prop` $prop^)? attr-dict"

    ctx = MLContext()
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
            '"test.optional_property"() <{"prop" = i32}> : () -> ()',
        ),
    ],
)
def test_optional_property_with_whitespace(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalPropertyOp(IRDLOperation):
        name = "test.optional_property"
        prop = opt_prop_def(Attribute)

        assembly_format = "`(` (` ` `prop` $prop^ ` `)? `)` attr-dict"

    ctx = MLContext()
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
            '"test.optional_unit_attr_prop"() <{"unit_attr"}> : () -> ()',
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

    ctx = MLContext()
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
            '"test.optional_unit_attr"() <{"unit_attr"}> : () -> ()',
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

    ctx = MLContext()
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
            '"test.optional_attribute"() {"attr" = i32} : () -> ()',
        ),
    ],
)
def test_optional_attribute(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class OptionalAttributeOp(IRDLOperation):
        name = "test.optional_attribute"
        attr = opt_attr_def(Attribute)

        assembly_format = "(`attr` $attr^)? attr-dict"

    ctx = MLContext()
    ctx.load_op(OptionalAttributeOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.typed_attr 3 3.000000e+00",
            '"test.typed_attr"() {"attr" = 3 : i32, "float_attr" = 3.000000e+00 : f64} : () -> ()',
        ),
    ],
)
def test_typed_attribute_variable(program: str, generic_program: str):
    """Test the parsing of typed attributes"""

    @irdl_op_definition
    class TypedAttributeOp(IRDLOperation):
        name = "test.typed_attr"
        attr = attr_def(IntegerAttr[I32])
        float_attr = attr_def(FloatAttr[Annotated[Float64Type, Float64Type()]])

        assembly_format = "$attr $float_attr attr-dict"

    ctx = MLContext()
    ctx.load_op(TypedAttributeOp)
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

    ctx = MLContext()
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
    ):

        @irdl_op_definition
        class UnknownVarOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.unknown_var_op"

            assembly_format = "$var attr-dict"


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
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.two_operands %0 %1 i32 i64",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.two_operands"(%0, %1) : (i32, i64) -> ()',
        ),
        (
            "$rhs $lhs type($rhs) type($lhs) attr-dict",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.two_operands %1 %0 i64 i32",
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

    ctx = MLContext()
    ctx.load_op(TwoOperandsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\n' "test.variadic_operand  ",
            '%0 = "test.op"() : () -> i32\n' '"test.variadic_operand"() : () -> ()',
        ),
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\n' "test.variadic_operand %0 i32",
            '%0 = "test.op"() : () -> i32\n'
            '"test.variadic_operand"(%0) : (i32) -> ()',
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

    ctx = MLContext()
    ctx.load_op(VariadicOperandOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\n' "test.optional_operand  ",
            '%0 = "test.op"() : () -> i32\n' '"test.optional_operand"() : () -> ()',
        ),
        (
            "$args type($args) attr-dict",
            '%0 = "test.op"() : () -> i32\n' "test.optional_operand %0 i32",
            '%0 = "test.op"() : () -> i32\n'
            '"test.optional_operand"(%0) : (i32) -> ()',
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

    ctx = MLContext()
    ctx.load_op(OptionalOperandOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program, as_property",
    [
        (
            '%0 = "test.op"() : () -> i32\n'
            "test.variadic_operands(%0 : i32) [%0 : i32]",
            '%0 = "test.op"() : () -> i32\n'
            '"test.variadic_operands"(%0, %0) {operandSegmentSizes = array<i32:1,1>} : (i32,i32) -> ()',
            False,
        ),
        (
            '%0 = "test.op"() : () -> i32\n'
            "test.variadic_operands(%0 : i32) [%0 : i32]",
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

    ctx = MLContext()
    ctx.load_op(VariadicOperandsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            "test.optional_operands(: ) [: ]",
            '"test.optional_operands"() {operandSegmentSizes = array<i32:0,0>} : () -> ()',
        ),
        (
            '%0 = "test.op"() : () -> i32\n'
            "test.optional_operands(%0 : i32) [%0 : i32]",
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

    ctx = MLContext()
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

    ctx = MLContext()
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
def test_operands_directive(program: str):
    """Test the operands directive"""

    @irdl_op_definition
    class OperandsDirectiveOp(IRDLOperation):
        name = "test.operands_directive"

        op1 = operand_def()
        op2 = var_operand_def()

        assembly_format = "operands `:` type(operands) attr-dict"

    ctx = MLContext()
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
    """Test the operands directive"""

    @irdl_op_definition
    class OperandsDirectiveOp(IRDLOperation):
        name = "test.operands_directive"

        op1 = opt_operand_def()
        op2 = operand_def()

        assembly_format = "operands `:` type(operands) attr-dict"

    ctx = MLContext()
    ctx.load_op(OperandsDirectiveOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)


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
        ("test.two_operands %0 : i32, i32", "Expected 2 operands but found 1"),
        (
            "test.two_operands %0, %1, %2 : i32, i32",
            "Expected 2 operands but found 3",
        ),
        ("test.two_operands %0, %1 : i32", "Expected 2 operand types but found 1"),
        (
            "test.two_operands %0, %1 : i32, i32, i32",
            "Expected 2 operand types but found 3",
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

    ctx = MLContext()
    ctx.load_op(TwoOperandsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "test.three_operands %0 : i32, i32",
            "Expected at least 2 operands but found 1",
        ),
        (
            "test.three_operands %0, %1, %2, %3 : i32, i32, i32",
            "Expected at most 3 operands but found 4",
        ),
        (
            "test.three_operands %0, %1 : i32",
            "Expected at least 2 operand types but found 1",
        ),
        (
            "test.three_operands %0, %1, %3 : i32, i32, i32, i32",
            "Expected at most 3 operand types but found 4",
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

    ctx = MLContext()
    ctx.load_op(ThreeOperandsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


@pytest.mark.parametrize(
    "program, error",
    [
        (
            "test.three_operands %0 : i32, i32",
            "Expected at least 2 operands but found 1",
        ),
        (
            "test.three_operands %0, %1 : i32",
            "Expected at least 2 operand types but found 1",
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

    ctx = MLContext()
    ctx.load_op(ThreeOperandsOp)

    with pytest.raises(ParseError, match=error):
        parser = Parser(ctx, program)
        parser.parse_operation()


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

    ctx = MLContext()
    ctx.load_op(TwoResultOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "`:` type($res) attr-dict",
            "test.variadic_result : ",
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

    ctx = MLContext()
    ctx.load_op(VariadicResultOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "`:` type($res) attr-dict",
            "test.optional_result : ",
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

    ctx = MLContext()
    ctx.load_op(OptionalResultOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


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
            'test.region_attr_dict {\n} {"a" = 2 : i32}',
            '"test.region_attr_dict"() ({}) {"a" = 2 : i32} : () -> ()',
        ),
        (
            "attr-dict `,` $region",
            'test.region_attr_dict {"a" = 2 : i32}, {\n  "test.op"() : () -> ()\n}',
            '"test.region_attr_dict"() ({  "test.op"() : () -> ()}) {"a" = 2 : i32} : () -> ()',
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

    ctx = MLContext()
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
            'test.two_regions attributes {"a" = 2 : i32} {\n  "test.op"() : () -> ()\n} {\n  "test.op"() : () -> ()\n}',
            '"test.two_regions"() ({ "test.op"() : () -> ()}, { "test.op"() : () -> ()}) {"a" = 2 : i32} : () -> ()',
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

    ctx = MLContext()
    ctx.load_op(TwoRegionsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "attr-dict-with-keyword $region",
            "test.variadic_region ",
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

    ctx = MLContext()
    ctx.load_op(VariadicRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "format, program, generic_program",
    [
        (
            "attr-dict-with-keyword $region",
            "test.optional_region ",
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

    ctx = MLContext()
    ctx.load_op(OptionalRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


def test_multiple_optional_regions():
    """Test that a variadic region variable cannot directly follow another variadic region variable."""
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="A variadic region variable cannot be followed by another variadic region variable.",
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
            "($opt_region^ `keyword`)? attr-dict",
            "test.optional_region_group",
            '"test.optional_region_group"() : () -> ()',
        ),
        (
            "($opt_region^ `keyword`)? attr-dict",
            'test.optional_region_group {\n  "test.op"() : () -> ()\n} keyword',
            '"test.optional_region_group"() ({"test.op"() : () -> ()}) : () -> ()',
        ),
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
        irdl_options = [AttrSizedRegionSegments]
        opt_region = opt_region_def()

        assembly_format = format

    ctx = MLContext()
    ctx.load_op(OptionalRegionOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


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


def test_successors():
    """Test the parsing of successors"""

    program = textwrap.dedent(
        """\
        "test.op"() ({
          "test.op"() [^0] : () -> ()
        ^0:
          test.two_successors ^0 ^0
        }) : () -> ()"""
    )

    generic_program = textwrap.dedent(
        """\
        "test.op"() ({
          "test.op"() [^0] : () -> ()
        ^0:
          "test.two_successors"() [^0, ^0] : () -> ()
        }) : () -> ()"""
    )

    @irdl_op_definition
    class TwoSuccessorsOp(IRDLOperation):
        name = "test.two_successors"
        fst = successor_def()
        snd = successor_def()

        assembly_format = "$fst $snd attr-dict"

    ctx = MLContext()
    ctx.load_op(TwoSuccessorsOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '"test.op"() ({\n  "test.op"() [^0] : () -> ()\n^0:\n  test.var_successor \n}) : () -> ()',
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  "test.var_successor"() : () -> ()
                }) : () -> ()"""
            ),
        ),
        (
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  test.var_successor ^0
                }) : () -> ()"""
            ),
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  "test.var_successor"() [^0] : () -> ()
                }) : () -> ()"""
            ),
        ),
        (
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  test.var_successor ^0 ^0
                }) : () -> ()"""
            ),
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  "test.var_successor"() [^0, ^0] : () -> ()
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

    ctx = MLContext()
    ctx.load_op(VarSuccessorOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '"test.op"() ({\n  "test.op"() [^0] : () -> ()\n^0:\n  test.opt_successor \n}) : () -> ()',
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  "test.opt_successor"() : () -> ()
                }) : () -> ()"""
            ),
        ),
        (
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  test.opt_successor ^0
                }) : () -> ()"""
            ),
            textwrap.dedent(
                """\
                "test.op"() ({
                  "test.op"() [^0] : () -> ()
                ^0:
                  "test.opt_successor"() [^0] : () -> ()
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

    ctx = MLContext()
    ctx.load_op(OptSuccessorOp)
    ctx.load_dialect(Test)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)


################################################################################
# Inference                                                                   #
################################################################################

_T = TypeVar("_T", bound=Attribute, covariant=True)
_ConstrT = TypeVar("_ConstrT", bound=Attribute, covariant=True)


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

    ctx = MLContext()
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

    ctx = MLContext()
    ctx.load_attr(UnitType)
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

    ctx = MLContext()
    ctx.load_attr(UnitType)
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

        n: ParameterDef[Attribute]
        p: ParameterDef[_T]
        q: ParameterDef[Attribute]

        @staticmethod
        def constr(
            *,
            n: GenericAttrConstraint[Attribute] | None = None,
            p: GenericAttrConstraint[_ConstrT] | None = None,
            q: GenericAttrConstraint[Attribute] | None = None,
        ) -> BaseAttr[ParamOne[Attribute]] | ParamAttrConstraint[ParamOne[_ConstrT]]:
            if n is None and p is None and q is None:
                return BaseAttr[ParamOne[Attribute]](ParamOne)
            return ParamAttrConstraint[ParamOne[_ConstrT]](ParamOne, (n, p, q))

    @irdl_op_definition
    class TwoOperandsNestedVarOp(IRDLOperation):
        T: ClassVar = VarConstraint("T", AnyAttr())

        name = "test.two_operands_one_result_with_var"
        res = result_def(T)
        lhs = operand_def(ParamOne.constr(p=T))
        rhs = operand_def(T)

        assembly_format = "$lhs $rhs attr-dict `:` type($lhs)"

    ctx = MLContext()
    ctx.load_op(TwoOperandsNestedVarOp)
    ctx.load_attr(ParamOne)
    ctx.load_dialect(Test)
    program = textwrap.dedent(
        """\
    %0, %1 = "test.op"() : () -> (!test.param_one<f16, i32, i1>, i32)
    %2 = test.two_operands_one_result_with_var %0 %1 : !test.param_one<f16, i32, i1>"""
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
        p: ParameterDef[_T]

        @staticmethod
        def constr(
            *,
            p: GenericAttrConstraint[_ConstrT] | None = None,
        ) -> BaseAttr[ParamOne[Attribute]] | ParamAttrConstraint[ParamOne[_ConstrT]]:
            if p is None:
                return BaseAttr[ParamOne[Attribute]](ParamOne)
            return ParamAttrConstraint[ParamOne[_ConstrT]](ParamOne, (p,))

    @irdl_op_definition
    class OneOperandOneResultNestedOp(IRDLOperation):
        T: ClassVar = VarConstraint("T", AnyAttr())

        name = "test.one_operand_one_result_nested"
        res = result_def(T)
        lhs = operand_def(ParamOne.constr(p=T))

        assembly_format = "$lhs attr-dict `:` type($lhs)"

    ctx = MLContext()
    ctx.load_op(OneOperandOneResultNestedOp)
    ctx.load_attr(ParamOne)
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
    class RangeVarOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
        name = "test.range_var"
        T: ClassVar = RangeVarConstraint("T", RangeOf(AnyAttr()))
        ins = var_operand_def(T)
        outs = var_result_def(T)

        assembly_format = "$ins attr-dict `:` type($ins)"

    with pytest.raises(
        NotImplementedError,
        match="Inference of length of variadic result 'outs' not implemented",
    ):
        ctx = MLContext()
        ctx.load_op(RangeVarOp)
        ctx.load_dialect(Test)
        program = textwrap.dedent("""\
        %in0, %in1 = "test.op"() : () -> (index, index)
        %out0, %out1 = test.range_var %in0, %in1 : index, index
        """)

        parser = Parser(ctx, program)
        test_op = parser.parse_optional_operation()
        assert isinstance(test_op, test.Operation)
        my_op = parser.parse_optional_operation()
        assert isinstance(my_op, RangeVarOp)


################################################################################
# Declarative Format Verification                                              #
################################################################################


@pytest.mark.parametrize(
    "variadic_def, format",
    [
        (var_operand_def, "$variadic `,` type($variadic) attr-dict"),
        (var_operand_def, "type($variadic) `,` attr-dict"),
        (var_result_def, "type($variadic) `,` attr-dict"),
        (opt_operand_def, "$variadic `,` type($variadic) attr-dict"),
        (opt_operand_def, "type($variadic) `,` attr-dict"),
        (opt_result_def, "type($variadic) `,` attr-dict"),
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
    "variadic_def_one",
    [var_operand_def, var_result_def, opt_operand_def, opt_result_def],
)
@pytest.mark.parametrize(
    "variadic_def_two",
    [var_operand_def, var_result_def, opt_operand_def, opt_result_def],
)
def test_chained_variadic_types_safeguard(
    variadic_def_one: Callable[[], VarOperand | VarOpResult],
    variadic_def_two: Callable[[], VarOperand | VarOpResult],
):
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="A variadic type directive cannot be followed by another variadic type directive.",
    ):

        @irdl_op_definition
        class VarTypeGuardOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
            name = "test.variadic_type_safeguard"

            variadic_one = variadic_def_one()
            variadic_two = variadic_def_two()
            assembly_format = "type($variadic_one) type($variadic_two) attr-dict"

            irdl_options = [AttrSizedOperandSegments(), AttrSizedResultSegments()]


@pytest.mark.parametrize("variadic_def_one", [var_operand_def, opt_operand_def])
@pytest.mark.parametrize("variadic_def_two", [var_operand_def, opt_operand_def])
def test_chained_variadic_operands_safeguard(
    variadic_def_one: Callable[[], VarOperand | VarOpResult],
    variadic_def_two: Callable[[], VarOperand | VarOpResult],
):
    with pytest.raises(
        PyRDLOpDefinitionError,
        match="A variadic operand variable cannot be followed by another variadic operand variable.",
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
            '%0 = "test.op"() : () -> i32\n' "test.optional_group(%0 : i32)",
            '%0 = "test.op"() : () -> i32\n' '"test.optional_group"(%0) : (i32) -> ()',
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

    ctx = MLContext()
    ctx.load_op(OptionalGroupOp)
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
            '%0 = "test.op"() : () -> i32\n' "test.optional_group %0 : i32",
            '%0 = "test.op"() : () -> i32\n' '"test.optional_group"(%0) : (i32) -> ()',
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

    ctx = MLContext()
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

    ctx = MLContext()
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

    ctx = MLContext()
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

            assembly_format = format


@pytest.mark.parametrize(
    "program, generic_program",
    [
        (
            '%0 = "test.op"() : () -> !test.type<"index">\n' "test.mixed %0()",
            '%0 = "test.op"() : () -> !test.type<"index">\n'
            '"test.mixed"(%0) : (!test.type<"index">) -> ()',
        ),
        (
            '%0 = "test.op"() : () -> !test.type<"index">\n' "test.mixed %0(%0)",
            '%0 = "test.op"() : () -> !test.type<"index">\n'
            '"test.mixed"(%0, %0) : (!test.type<"index">, !test.type<"index">) -> ()',
        ),
        (
            '%0 = "test.op"() : () -> !test.type<"index">\n' "test.mixed %0(%0, %0)",
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

    ctx = MLContext()
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
            '"test.default"() <{"prop" = false}> {"attr" = false} : () -> ()',
        ),
        (
            "test.default prop false opt_prop true",
            "test.default",
            '"test.default"() <{"prop" = false, "opt_prop" = true}> {"attr" = false} : () -> ()',
        ),
        (
            '"test.default"() <{"prop" = false, "opt_prop" = true}> {"attr" = false} : () -> ()',
            "test.default",
            '"test.default"() <{"prop" = false, "opt_prop" = true}> {"attr" = false} : () -> ()',
        ),
        (
            '"test.default"() <{"prop" = false}> {"attr" = false} : () -> ()',
            "test.default",
            '"test.default"() <{"prop" = false}> {"attr" = false} : () -> ()',
        ),
        (
            "test.default prop true opt_prop false",
            "test.default prop 1 opt_prop 0",
            '"test.default"() <{"prop" = true, "opt_prop" = false}> {"attr" = false} : () -> ()',
        ),
        (
            "test.default attr false opt_attr true",
            "test.default",
            '"test.default"() <{"prop" = false}> {"attr" = false, "opt_attr" = true} : () -> ()',
        ),
        (
            '"test.default"() <{"prop" = false}> {"attr" = false, "opt_attr" = true} : () -> ()',
            "test.default",
            '"test.default"() <{"prop" = false}> {"attr" = false, "opt_attr" = true} : () -> ()',
        ),
        (
            "test.default attr true opt_attr false",
            "test.default attr 1 opt_attr 0",
            '"test.default"() <{"prop" = false}> {"attr" = true, "opt_attr" = false} : () -> ()',
        ),
        (
            '"test.default"() : () -> ()',
            "test.default",
            '"test.default"() <{"prop" = false}> {"attr" = false} : () -> ()',
        ),
    ],
)
def test_default_properties(program: str, output: str, generic: str):
    ctx = MLContext()
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
            '"test.renamed"() <{"test_prop1" = false}> : () -> ()',
        ),
        (
            "test.renamed prop1 false prop2 false",
            "test.renamed prop2 0",
            '"test.renamed"() <{"test_prop1" = false, "test_prop2" = false}> : () -> ()',
        ),
        (
            "test.renamed prop1 true prop2 true",
            "test.renamed prop1 1 prop2 1",
            '"test.renamed"() <{"test_prop1" = true, "test_prop2" = true}> : () -> ()',
        ),
    ],
)
def test_renamed_optional_prop(program: str, output: str, generic: str):
    ctx = MLContext()
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


################################################################################
#                                Extractors                                    #
################################################################################


@irdl_op_definition
class AllOfExtractorOp(IRDLOperation):
    name = "test.all_of_extractor"

    T: ClassVar = VarConstraint("T", AnyAttr())
    lhs = operand_def(T & MemRefType.constr(element_type=T))
    rhs = operand_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"


def test_all_of_extraction_fails():
    ctx = MLContext()
    ctx.load_op(AllOfExtractorOp)
    ctx.load_dialect(Test)
    parser = Parser(
        ctx,
        '%0 = "test.op"() : () -> memref<10xindex>\ntest.all_of_extractor %0, %0 : memref<10xindex>',
    )
    parser.parse_operation()
    with pytest.raises(
        ValueError,
        match="Value of variable T could not be uniquely extracted.\n"
        "Possible values are: {index, memref<10xindex>}",
    ):
        parser.parse_operation()


@irdl_attr_definition
class DoubleParamAttr(ParametrizedAttribute, TypeAttribute):
    """An attribute with two unbounded attribute parameters."""

    name = "test.param"

    param1: ParameterDef[Attribute]
    param2: ParameterDef[Attribute]


@irdl_op_definition
class ParamExtractorOp(IRDLOperation):
    name = "test.param_extractor"

    T: ClassVar = VarConstraint("T", AnyAttr())
    lhs = operand_def(ParamAttrConstraint(DoubleParamAttr, (T, T)))
    rhs = operand_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"


def test_param_extraction_fails():
    ctx = MLContext()
    ctx.load_attr(DoubleParamAttr)
    ctx.load_op(ParamExtractorOp)
    ctx.load_dialect(Test)
    parser = Parser(
        ctx,
        '%0 = "test.op"() : () -> !test.param<i32,i64>\ntest.param_extractor %0, %0 : !test.param<i32,i64>',
    )
    parser.parse_operation()
    with pytest.raises(
        ValueError,
        match="Value of variable T could not be uniquely extracted.\n"
        "Possible values are: {i32, i64}",
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
    ctx = MLContext()
    ctx.load_op(MultipleOperandExtractorOp)
    ctx.load_dialect(Test)
    parser = Parser(
        ctx,
        '%0, %1 = "test.op"() : () -> (index, i32)\ntest.multiple_operand_extractor %0, %1 : index, i32',
    )
    parser.parse_operation()
    with pytest.raises(
        ValueError,
        match="Value of variable T could not be uniquely extracted.\n"
        "Possible values are: {i32, index}",
    ):
        parser.parse_operation()
