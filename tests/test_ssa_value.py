import re

import pytest

from xdsl.dialects.builtin import StringAttr, i32, i64
from xdsl.ir import Block, BlockArgument, Operation, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.rewriter import Rewriter
from xdsl.utils.test_value import create_ssa_value


@irdl_op_definition
class TwoResultOp(IRDLOperation):
    name = "test.tworesults"

    res1 = result_def(StringAttr)
    res2 = result_def(StringAttr)


def test_var_mixed_builder():
    op = TwoResultOp.build(result_types=[StringAttr("0"), StringAttr("2")])

    with pytest.raises(
        ValueError, match="SSAValue.get: expected operation with a single result."
    ):
        _ = SSAValue.get(op)


@pytest.mark.parametrize(
    "name",
    [
        "test",
        "-2",
        "test123",
        "kebab-case-name",
        None,
    ],
)
def test_ssa_value_name_hints(name: str | None):
    r"""
    As per the MLIR language reference, legal SSA value names must conform to
        ([0-9]+|([A-Za-z_$.-][\w$.-]*))

    https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords

    xDSL SSA value name hints are a refinement of these rules.
    We only accept non-numeric name hints, because the printer will
    generate its own numeric names.

    This test tests valid name hints:
    """
    val = BlockArgument(i32, Block(), 0)

    val.name_hint = name
    assert val.name_hint == name


@pytest.mark.parametrize("name", ["&", "#", "%2", '"', "::", "42"])
def test_invalid_ssa_vals(name: str):
    """
    This test tests invalid name hints that raise an error, because
    they don't conform to the rules of how SSA value names should be
    structured.
    """
    val = BlockArgument(i32, Block(), 0)
    with pytest.raises(
        ValueError, match=re.escape(f"Invalid BlockArgument name format `{name}`.")
    ):
        val.name_hint = name


def test_extract_valid_name():
    """
    This test tests invalid name hints that raise an error, because
    they don't conform to the rules of how SSA value names should be
    structured.
    """
    assert SSAValue.extract_valid_name("hello") == "hello"
    assert SSAValue.extract_valid_name("hello_1") == "hello"
    with pytest.raises(ValueError, match="Invalid SSAValue name format"):
        SSAValue.extract_valid_name("1hello")


def test_rewrite_type():
    """We can rewrite the type of a test SSA value."""
    val = create_ssa_value(i32)
    rewriter = Rewriter()
    new_val = rewriter.replace_value_with_new_type(val, i64)
    assert val.type == i32
    assert new_val.type == i64


def test_replace_type():
    class InvalidValue(SSAValue):
        @property
        def owner(self) -> Operation | Block:
            """
            An SSA variable is either an operation result, or a basic block argument.
            This property returns the Operation or Block that currently defines a specific value.
            """
            ...

    val = InvalidValue(i32)

    rewriter = Rewriter()

    with pytest.raises(
        ValueError,
        match="Expected OpResult or BlockArgument, got InvalidValue",
    ):
        rewriter.replace_value_with_new_type(val, i32)
