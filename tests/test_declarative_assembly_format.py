from __future__ import annotations

from io import StringIO

import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import MLContext, Operation
from xdsl.irdl import IRDLOperation, irdl_op_definition
from xdsl.parser.core import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLOpDefinitionError

################################################################################
# Utils for this test file                                                     #
################################################################################


def check_roundtrip(program: str, ctx: MLContext):
    """Check that the given program roundtrips exactly (including whitespaces)."""
    parser = Parser(ctx, program)
    ops: list[Operation] = []
    while op := parser.parse_optional_operation():
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
    while op := parser.parse_optional_operation():
        ops1.append(op)

    parser = Parser(ctx, program2)
    ops2: list[Operation] = []
    while op := parser.parse_optional_operation():
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
        class FormatAndPrintOp(IRDLOperation):  # type: ignore[reportUnusedImport]
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
        class FormatAndParseOp(IRDLOperation):  # type: ignore[reportUnusedImport]
            name = "test.format_and_parse"

            assembly_format = "attr-dict"

            @classmethod
            def parse(cls, parser: Parser) -> FormatAndParseOp:
                raise NotImplementedError()

    with pytest.raises(
        PyRDLOpDefinitionError, match="Cannot define both an assembly format"
    ):

        @irdl_op_definition
        class FormatAndParseOp(IRDLOperation):  # type: ignore[reportUnusedImport]
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
        class NoAttrDictOp(IRDLOperation):  # type: ignore[reportUnusedImport]
            name = "test.no_attr_dict"

            assembly_format = ""


def test_two_attr_dicts():
    """Check that we cannot have two attr-dicts."""

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive has already been seen"
    ):

        @irdl_op_definition
        class NoAttrDictOp(IRDLOperation):  # type: ignore[reportUnusedImport]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive has already been seen"
    ):

        @irdl_op_definition
        class NoAttrDictOp(IRDLOperation):  # type: ignore[reportUnusedImport]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict attr-dict-with-keyword"

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive has already been seen"
    ):

        @irdl_op_definition
        class NoAttrDictOp(IRDLOperation):  # type: ignore[reportUnusedImport]
            name = "test.no_attr_dict"

            assembly_format = "attr-dict-with-keyword attr-dict"

    with pytest.raises(
        PyRDLOpDefinitionError, match="'attr-dict' directive has already been seen"
    ):

        @irdl_op_definition
        class NoAttrDictOp(IRDLOperation):  # type: ignore[reportUnusedImport]
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
    ctx.register_op(AttrDictOp)
    ctx.register_op(AttrDictWithKeywordOp)

    check_roundtrip(program, ctx)
    check_equivalence(program, generic_program, ctx)
