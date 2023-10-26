from __future__ import annotations

from io import StringIO

import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.test import Test
from xdsl.ir import MLContext, Operation
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLOpDefinitionError

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

    assert ModuleOp(ops1).is_structurally_equivalent(ModuleOp(ops2))


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


################################################################################
# Punctuations and keywords                                                    #
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
        PyRDLOpDefinitionError, match="type of operand 'operand' not found"
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
    with pytest.raises(PyRDLOpDefinitionError, match="result 'result' not found"):

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
