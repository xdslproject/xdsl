from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.dialects.test import TestOp
from xdsl.folder import Folder
from xdsl.ir import Block, OpResult
from xdsl.rewriter import InsertPoint


def test_try_fold_foldable_operation():
    """Test that try_fold correctly folds an AddiOp with zero."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    # Create constants: 0 and 5
    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)

    # Create an AddiOp: %result = arith.addi %zero, %five : i32
    # Adding zero should fold to just the other operand
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

    # Create constants: 0 and 5
    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)
    builder.insert(one_const)
    builder.insert(five_const)

    # Create an AddiOp: %result = arith.addi %zero, %five : i32
    # Adding zero should fold to just the other operand
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


def test_replace_with_fold():
    """Test that replace_with_fold correctly folds an AddiOp with zero."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    # Create constants: 0 and 5
    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)
    builder.insert(one_const)
    builder.insert(five_const)

    # Create an AddiOp: %result = arith.addi %zero, %five : i32
    # Adding zero should fold to just the other operand
    addi_op = AddiOp(one_const.result, five_const.result)
    builder.insert(addi_op)

    # Replace with fold
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
