from xdsl.dialects import test
from xdsl.dialects.arith import Constant
from xdsl.builder import Builder
from xdsl.dialects.builtin import IntegerAttr, i32, IndexType
from xdsl.ir import BlockArgument
from xdsl.dialects.scf import For, Yield
from xdsl.ir import Block
from xdsl.dialects.arith import Addi, Constant, DivUI, Muli
from xdsl.transforms.loop_invariant_code_motion import (
    can_Be_Hoisted
)

index = IndexType()

def test_is_loop_dependent_no_dep():
    lb = Constant.from_int_and_width(0, i32)
    ub = Constant.from_int_and_width(42, i32)
    step = Constant.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])

    for_op = For(lb, ub, step, [], Block([op1, op2], arg_types=[i32]))

    assert can_Be_Hoisted(op1, for_op.body)
    assert can_Be_Hoisted(op2, for_op.body)

def test_for_with_loop_invariant_verify():
    """Test for with loop-variant variables"""

    lower = Constant.from_int_and_width(0, IndexType())
    upper = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())

    a = Constant(IntegerAttr.from_int_and_width(1, 32), i32)
    b = Constant(IntegerAttr.from_int_and_width(2, 32), i32)

    @Builder.implicit_region((IndexType(), IndexType()))
    def body(_: tuple[BlockArgument, ...]) -> None:
        # Operations on these constants
        c = Addi(a, b)
        d = Muli(a, b)
        e = DivUI(c, d)
        Yield(e)

    for_op = For(lower, upper, step, [carried], body)

    for op in for_op.body.walk():
        for regoin in for_op.regions:
            #MLIR iterates the for loop's body and hoists operations that can_be_hoisted. However, the DivUI operation cannot be hoisted immediately because its operands are still within the loop's body. It can only be hoisted once the instructions generating those operands have been hoisted.
            if not (isinstance(op, Yield) or isinstance(op, DivUI)):
                assert can_Be_Hoisted(op, regoin)

def test_for_with_loop_invariant_verify1():
    """Test for with loop-invariant variables"""
    lb = Constant.from_int_and_width(0, i32)
    ub = Constant.from_int_and_width(42, i32)
    step = Constant.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp([op2, op1], result_types=[i32])
    op4 = test.TestOp([op3, op2], result_types=[i32])
    bb0 = Block([op1, op2, op3, op4], arg_types=[i32])

    for_op = For(lb, ub, step, [], bb0)

    for region in for_op.regions:
        assert can_Be_Hoisted(op1, region)
        assert can_Be_Hoisted(op2, region)
        assert not can_Be_Hoisted(op3, region)
        assert not can_Be_Hoisted(op4, region)

