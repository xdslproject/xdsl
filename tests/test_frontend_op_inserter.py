import pytest

from xdsl.dialects.affine import For
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.builtin import i32
from xdsl.frontend.exception import FrontendProgramException
from xdsl.frontend.op_inserter import OpInserter
from xdsl.ir import Block, Region


def test_raises_exception_on_empty_stack():
    inserter = OpInserter(Block())
    with pytest.raises(FrontendProgramException) as err:
        inserter.get_operand()
    assert err.value.msg == "Trying to get an operand from an empty stack."


def test_raises_exception_on_op_with_no_regions():
    inserter = OpInserter(Block())
    op_with_no_region = Constant.from_int_and_width(1, i32)
    with pytest.raises(FrontendProgramException) as err:
        inserter.set_insertion_point_from_op(op_with_no_region)
    assert err.value.msg == (
        "Trying to set the insertion point for operation"
        " 'arith.constant' with no regions."
    )


def test_raises_exception_on_op_with_no_blocks():
    inserter = OpInserter(Block())
    op_with_no_region = For.from_region([], [], 0, 10, Region())
    with pytest.raises(FrontendProgramException) as err:
        inserter.set_insertion_point_from_op(op_with_no_region)
    assert err.value.msg == (
        "Trying to set the insertion point for operation"
        " 'affine.for' with no blocks in its last region."
    )


def test_raises_exception_on_op_with_no_blocks_II():
    inserter = OpInserter(Block())
    empty_region = Region()
    with pytest.raises(FrontendProgramException) as err:
        inserter.set_insertion_point_from_region(empty_region)
    assert err.value.msg == (
        "Trying to set the insertion point from the region without" " blocks."
    )


def test_inserts_ops():
    region = Region([Block(), Block()])
    inserter = OpInserter(region.blocks[0])

    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)

    inserter.insert_op(a)
    inserter.insert_op(b)

    b = inserter.get_operand()
    a = inserter.get_operand()

    c = Addi(a, b)
    inserter.insert_op(c)

    assert len(inserter.stack) == 1
    assert inserter.stack[0] is c.results[0]
