import pytest

from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException


def test_func():
    # This test creates two FuncOps with different approaches that
    # represent the same code and checks their structure
    # Create two constants and add them, add them in a region and
    # create a function
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    # Create a region to include a, b, c
    region = Region.from_operation_list([a, b, c])

    # Use this region to create a func0
    func0 = FuncOp.from_region("func0", [], [], region)

    # Alternative generation of func0
    func1 = FuncOp.from_region(
        "func1",
        [],
        [],
        Region.from_operation_list([
            a := Constant.from_int_and_width(1, i32),
            b := Constant.from_int_and_width(2, i32),
            Addi.get(a, b)
        ]))   # yapf: disable
    # yapf disabled for structured look of this test

    assert len(func0.regions[0].ops) == 3
    assert len(func1.regions[0].ops) == 3
    assert type(func0.regions[0].ops[0]) is Constant
    assert type(func1.regions[0].ops[0]) is Constant
    assert type(func0.regions[0].ops[1]) is Constant
    assert type(func1.regions[0].ops[1]) is Constant
    assert type(func0.regions[0].ops[2]) is Addi


def test_func_II():
    # Create constants `from_attr` and add them, add them in blocks, blocks in
    # a region and create a function
    a = Constant.from_attr(IntegerAttr.from_int_and_width(1, 32), i32)
    b = Constant.from_attr(IntegerAttr.from_int_and_width(2, 32), i32)
    c = Constant.from_attr(IntegerAttr.from_int_and_width(3, 32), i32)
    d = Constant.from_attr(IntegerAttr.from_int_and_width(4, 32), i32)

    # Operation to add these constants
    e = Addi.get(a, b)
    f = Addi.get(c, d)

    # Create Blocks and Regions
    block0 = Block.from_ops([a, b, e])
    block1 = Block.from_ops([c, d, f])
    region0 = Region.from_block_list([block0, block1])

    # Use this region to create a func0
    func1 = FuncOp.from_region("func1", [], [], region0)

    assert len(func1.regions[0].blocks[0].ops) == 3
    assert len(func1.regions[0].blocks[1].ops) == 3
    assert type(func1.regions[0].blocks[0].ops[0]) is Constant
    assert type(func1.regions[0].blocks[0].ops[1]) is Constant
    assert type(func1.regions[0].blocks[0].ops[2]) is Addi
    assert type(func1.regions[0].blocks[1].ops[0]) is Constant
    assert type(func1.regions[0].blocks[1].ops[1]) is Constant
    assert type(func1.regions[0].blocks[1].ops[2]) is Addi


def test_wrong_blockarg_types():
    r = Region.from_block_list(
        [Block.from_callable([i32], lambda x: [Addi.get(x, x)])])
    f = FuncOp.from_region("f", [i32, i32], [], r)
    with pytest.raises(VerifyException) as e:
        f.verify()

    assert e.value.args[
        0] == "Expected entry block arguments to have the same types as the function input types"


def test_callable_constructor():
    f = FuncOp.from_callable("f", [], [], lambda: [])
    assert f.sym_name.data == "f"
    assert f.body.ops == []


def test_return():
    # Create two constants and add them, then return
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    c = Constant.from_int_and_width(3, i32)

    # Use these region to create a func0
    ret0 = Return.get(a, b, c)
    assert len(ret0.operands) == 3
