import pytest

from xdsl.dialects import test
from xdsl.dialects.pdl import AttributeType, RangeType, TypeType

# Assuming the previous code is in a module named 'pdl_interp_ext'
from xdsl.dialects.pdl_interp import ContinueOp, ForEachOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException


def test_foreach_valid():
    """Test a correctly constructed ForEachOp."""
    val = test.TestOp(result_types=(RangeType(TypeType()),)).results[0]

    foreach_block = Block()
    successor_block = Block()

    # Region block argument type (!pdl.type) matches range element type
    body_block = Block(arg_types=[TypeType()])
    body_block.add_op(ContinueOp())

    op = ForEachOp(val, successor_block, Region([body_block]))
    foreach_block.add_op(op)
    _parent_region = Region([foreach_block, successor_block])

    # Should not raise
    op.verify()


def test_foreach_argument_mismatch():
    """Test that region argument type must match range element type."""
    val = test.TestOp(result_types=(RangeType(TypeType()),)).results[0]

    forach_block = Block()
    successor_block = Block()

    # Region argument is !pdl.attribute, but range contains !pdl.type
    body_block = Block(arg_types=[AttributeType()])
    body_block.add_op(ContinueOp())

    op = ForEachOp(val, successor_block, Region([body_block]))
    forach_block.add_op(op)
    _parent_region = Region([forach_block, successor_block])

    with pytest.raises(VerifyException, match="Region argument type .* does not match"):
        op.verify()


def test_foreach_invalid_arg_count():
    """Test that the region block must have exactly one argument."""
    val = test.TestOp(result_types=(RangeType(TypeType()),)).results[0]

    foreach_block = Block()
    successor_block = Block()

    # Block has 0 arguments
    body_block = Block(arg_types=[])
    body_block.add_op(ContinueOp())

    op = ForEachOp(val, successor_block, Region([body_block]))
    foreach_block.add_op(op)
    _parent_region = Region([foreach_block, successor_block])

    with pytest.raises(VerifyException, match="Region must have exactly one argument"):
        op.verify()


def test_foreach_empty_region():
    """Test that the region cannot be empty."""
    val = test.TestOp(result_types=(RangeType(TypeType()),)).results[0]

    foreach_block = Block()
    successor_block = Block()

    op = ForEachOp(val, successor_block, Region())
    foreach_block.add_op(op)
    _parent_region = Region([foreach_block, successor_block])

    with pytest.raises(VerifyException, match="Region must not be empty"):
        op.verify()
