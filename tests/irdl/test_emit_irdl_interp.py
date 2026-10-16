"""
Tests for emit_irdl_interp functionality.
"""

from xdsl.builder import Builder
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.irdl_interp import AssertOp, CheckBaseOp, ConstantOp
from xdsl.ir import Block
from xdsl.irdl import AnyAttr, BaseAttr
from xdsl.rewriter import InsertPoint


def test_emit_base_attr_irdl_interp():
    """Test that BaseAttr constraint lowering generates correct IR."""
    # Create a module to build in
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)

    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with
    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Emit irdl_interp IR
    constraint = BaseAttr(IntegerAttr)
    constraint.emit_irdl_interp(constant_op.result, builder)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: ConstantOp, CheckBaseOp, AssertOp
    assert len(ops) == 3

    # Check that CheckBaseOp was created
    check_op = ops[1]
    assert isinstance(check_op, CheckBaseOp)
    assert check_op.attr is constant_op.result
    assert check_op.base.data == "integer"

    # Check that AssertOp was created
    assert_op = ops[2]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is check_op.result
    assert "Expected integer" in assert_op.message.data


def test_any_attr_emit_is_noop():
    """Test that AnyAttr.emit_irdl_interp is a no-op."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Count ops before
    ops_before = len(list(block.ops))

    # Call emit_irdl_interp - should be a no-op
    any_attr = AnyAttr()
    any_attr.emit_irdl_interp(constant_op.result, builder)

    # Count ops after - should be the same
    ops_after = len(list(block.ops))
    assert ops_before == ops_after, (
        "AnyAttr.emit_irdl_interp should not generate any IR"
    )
