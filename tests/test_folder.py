import pytest

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.dialects.test import TestOp
from xdsl.folder import Folder
from xdsl.ir import Block, Operation, OpResult
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import HasFolder


def test_try_fold_foldable_operation():
    """Test that try_fold correctly folds an AddiOp with two constants."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)

    # Adding two constants should fold to a constant operation
    addi_op = AddiOp(one_const.result, five_const.result)

    # Try to fold the operation
    folder = Folder(ctx)
    result = folder.try_fold(addi_op)

    # Should successfully fold
    assert result is not None
    values, new_ops = result
    assert len(values) == 1
    assert len(new_ops) == 1
    assert isinstance(new_op := new_ops[0], ConstantOp)
    assert isinstance(new_op.value, IntegerAttr)
    assert new_op.value.value.data == 6


def test_insert_with_fold():
    """Test that insert_with_fold correctly folds an AddiOp with zero."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)
    builder.insert(one_const)
    builder.insert(five_const)

    addi_op = AddiOp(one_const.result, five_const.result)

    # Insert with fold
    folder = Folder(ctx)
    folded_values = folder.insert_with_fold(addi_op, builder)

    # Should successfully fold
    assert folded_values is not None
    assert len(folded_values) == 1
    assert isinstance(folded_value := folded_values[0], OpResult)
    defining_op = folded_value.owner
    assert isinstance(defining_op, ConstantOp)
    assert isinstance(defining_op.value, IntegerAttr)
    assert defining_op.value.value.data == 6
    assert defining_op.parent is block


def test_insert_with_fold_already_inserted():
    """Test that insert_with_fold raises an error when trying to fold an already inserted operation."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)
    builder.insert(one_const)
    builder.insert(five_const)

    addi_op = AddiOp(one_const.result, five_const.result)
    builder.insert(addi_op)

    # Insert with fold
    folder = Folder(ctx)
    msg = "Can't insert_with_fold fold an operation that already has a parent."
    with pytest.raises(ValueError, match=msg):
        folder.insert_with_fold(addi_op, builder)


def test_replace_with_fold():
    """Test that replace_with_fold correctly folds an AddiOp with zero."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)
    builder.insert(one_const)
    builder.insert(five_const)

    addi_op = AddiOp(one_const.result, five_const.result)
    builder.insert(addi_op)
    rewriter = PatternRewriter(addi_op)

    # Replace with fold
    folder = Folder(ctx)
    folded_values = folder.replace_with_fold(addi_op, rewriter)

    # Should successfully fold
    assert folded_values is not None
    assert len(folded_values) == 1
    assert isinstance(folded_value := folded_values[0], OpResult)
    defining_op = folded_value.owner
    assert isinstance(defining_op, ConstantOp)
    assert isinstance(defining_op.value, IntegerAttr)
    assert defining_op.value.value.data == 6
    assert defining_op.parent is block


def test_try_fold_unfoldable_operation():
    """Test that try_fold returns None for operations that don't support folding."""
    ctx = Context()

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    # Create a TestOp which doesn't implement folding
    const_op = ConstantOp.from_int_and_width(5, i32)
    builder.insert(const_op)
    test_op = TestOp((const_op.result,), result_types=(i32,))
    builder.insert(test_op)

    # Try to fold the TestOp
    folder = Folder(ctx)
    result = folder.try_fold(test_op)

    # Should return None since TestOp doesn't implement folding
    assert result is None


def test_fold_dynamic_trait():
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    @irdl_op_definition
    class TestFoldOp(IRDLOperation):
        name = "arith.fold"
        res = result_def()

    class TestFold(HasFolder):
        @classmethod
        def fold(cls, op: Operation):
            """
            Attempts to fold the operation. The fold method cannot modify the IR.
            Returns either an existing SSAValue or an Attribute for each result of the operation.
            When folding is unsuccessful, returns None.
            """
            assert isinstance(op, TestFoldOp)
            return (IntegerAttr(1, i32),)

    folder = Folder(ctx)
    op = TestFoldOp(result_types=(i32,))

    assert folder.try_fold(op) is None

    TestFoldOp.traits.add_trait(TestFold())

    assert folder.try_fold(op) is not None
