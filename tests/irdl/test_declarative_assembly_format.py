from __future__ import annotations

import textwrap
from collections.abc import Callable
from io import StringIO
from typing import Annotated, Generic, TypeVar

import pytest

from xdsl.context import MLContext
from xdsl.dialects.builtin import I32, IntegerAttr, ModuleOp, UnitAttr
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
    AttrSizedResultSegments,
    ConstraintVar,
    EqAttrConstraint,
    IRDLOperation,
    ParameterDef,
    ParsePropInAttrDict,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, PyRDLOpDefinitionError

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
        class FormatAndParseOp(IRDLOperation):  # pyright: ignore[reportUnusedClass]
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
class OpWithAttr(IRDLOperation):
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
    ctx.load_op(OpWithAttr)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


def test_attr_variable_shadowed():
    ctx = MLContext()
    ctx.load_op(OpWithAttr)

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
    class OpWithRenamedAttr(IRDLOperation):
        name = "test.one_attr"

        python = attr_def(Attribute, attr_name="irdl")
        assembly_format = "$irdl attr-dict"

    ctx = MLContext()
    ctx.load_op(OpWithRenamedAttr)

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
    class OpWithUnqualifiedAttr(IRDLOperation):
        name = "test.one_attr"

        attr = attr_def(ParamOne)
        assembly_format = "$attr attr-dict"

    ctx = MLContext()
    ctx.load_attr(ParamOne)
    ctx.load_op(OpWithUnqualifiedAttr)

    check_equivalence(program, generic_program, ctx)
    check_roundtrip(program, ctx)


def test_missing_property_error():
    class OpWithMissingProp(IRDLOperation):
        name = "test.missing_prop"

        prop1 = prop_def(Attribute)
        prop2 = prop_def(Attribute)
        assembly_format = "$prop1 attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError,
        match="prop2 properties are missing",
    ):
        irdl_op_definition(OpWithMissingProp)


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
    class OpWithProp(IRDLOperation):
        name = "test.one_prop"

        prop = prop_def(Attribute)
        assembly_format = "$prop attr-dict"

    ctx = MLContext()
    ctx.load_op(OpWithProp)

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
    class OpWithRenamedProp(IRDLOperation):
        name = "test.one_prop"

        python = prop_def(Attribute, prop_name="irdl")
        assembly_format = "$irdl attr-dict"

    ctx = MLContext()
    ctx.load_op(OpWithRenamedProp)

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
            "test.optional_unit_attr_prop",
            '"test.optional_unit_attr_prop"() : () -> ()',
        ),
        (
            "test.optional_unit_attr_prop unit_attr ",  # todo this prints an extra whitespace
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
            "test.optional_unit_attr unit_attr ",  # todo this prints an extra whitespace
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
            "test.typed_attr 3",
            '"test.typed_attr"() {"attr" = 3 : i32} : () -> ()',
        ),
    ],
)
def test_typed_attribute_variable(program: str, generic_program: str):
    """Test the parsing of optional operands"""

    @irdl_op_definition
    class TypedAttributeOp(IRDLOperation):
        name = "test.typed_attr"
        attr = attr_def(IntegerAttr[I32])

        assembly_format = "$attr attr-dict"

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
        match="expected variable to refer to an operand, attribute, region, result, or successor",
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
    "program, generic_program",
    [
        (
            '%0 = "test.op"() : () -> i32\n'
            "test.variadic_operands(%0 : i32) [%0 : i32]",
            '%0 = "test.op"() : () -> i32\n'
            '"test.variadic_operands"(%0, %0) {operandSegmentSizes = array<i32:1,1>} : (i32,i32) -> ()',
        ),
        (
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            "test.variadic_operands(%0, %1 : i32, i64) [%1, %0 : i64, i32]",
            '%0, %1 = "test.op"() : () -> (i32, i64)\n'
            '"test.variadic_operands"(%0, %1, %1, %0) {operandSegmentSizes = array<i32:2,2>} : (i32, i64, i64, i32) -> ()',
        ),
        (
            '%0, %1, %2 = "test.op"() : () -> (i32, i64, i128)\n'
            "test.variadic_operands(%0, %1, %2 : i32, i64, i128) [%2, %1, %0 : i128, i64, i32]",
            '%0, %1, %2 = "test.op"() : () -> (i32, i64, i128)\n'
            '"test.variadic_operands"(%0, %1, %2, %2, %1, %0) {operandSegmentSizes = array<i32:3,3>} : (i32, i64, i128, i128, i64, i32) -> ()',
        ),
    ],
)
def test_multiple_variadic_operands(program: str, generic_program: str):
    """Test the parsing of variadic operands"""

    @irdl_op_definition
    class VariadicOperandsOp(IRDLOperation):
        name = "test.variadic_operands"
        args1 = var_operand_def()
        args2 = var_operand_def()

        irdl_options = [AttrSizedOperandSegments()]

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
# Inference                                                                   #
################################################################################

_T = TypeVar("_T", bound=Attribute)


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
        T = Annotated[Attribute, ConstraintVar("T")]

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
    class OneOperandEqType(IRDLOperation):
        name = "test.one_operand_eq_type"
        index = operand_def(UnitType())
        res = result_def(UnitType())

        assembly_format = "attr-dict $index"

    ctx = MLContext()
    ctx.load_attr(UnitType)
    ctx.load_op(OneOperandEqType)
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
    class OneOperandEqTypeAllOfNested(IRDLOperation):
        name = "test.one_operand_eq_type_all_of_nested"
        index = operand_def(AllOf((AnyAttr(), EqAttrConstraint(UnitType()))))

        assembly_format = "attr-dict $index"

    ctx = MLContext()
    ctx.load_attr(UnitType)
    ctx.load_op(OneOperandEqTypeAllOfNested)
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

    @irdl_op_definition
    class TwoOperandsNestedVarOp(IRDLOperation):
        T = Annotated[Attribute, ConstraintVar("T")]

        name = "test.two_operands_one_result_with_var"
        res = result_def(T)
        lhs = operand_def(ParamOne[T])
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

    @irdl_op_definition
    class OneOperandOneResultNestedOp(IRDLOperation):
        T = Annotated[Attribute, ConstraintVar("T")]

        name = "test.one_operand_one_result_nested"
        res = result_def(T)
        lhs = operand_def(ParamOne[T])

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
        ParseError,
        match="Verification error while inferring operation type: ",
    ):
        check_roundtrip(program, ctx)


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
    (
        ("()?", "An optional group cannot be empty"),
        ("(`keyword`)?", "Every optional group must have an anchor."),
        (
            "($args^ type($rets)^)?",
            "An optional group can only have one anchor.",
        ),
        ("(`keyword`^)?", "An optional group's anchor must be an achorable directive."),
        (
            "($mandatory_arg^)?",
            "First element of an optional group must be optionally parsable.",
        ),
    ),
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
