import pytest
from __future__ import annotations

from xdsl.dialects.func import FuncOp
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.ir import Block, Region

from xdsl.dialects.builtin import IntegerAttr, IndexType
from xdsl.ir import OpResult


from xdsl.printer import Printer
from xdsl.ir import OpResult
from xdsl.dialects.builtin import i32, IntegerType, IndexType
from xdsl.dialects.memref import (Alloc, Alloca, Dealloc, Dealloca, MemRefType,
                                  Load, Store)


def test_matmul_func():

    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    A = OpResult(i32_memref_type, [], [])
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
    func1 = FuncOp.from_region("matmul", [], [], region0)

    assert len(func1.regions[0].blocks[0].ops) == 3
    assert len(func1.regions[0].blocks[1].ops) == 3
    assert type(func1.regions[0].blocks[0].ops[0]) is Constant
    assert type(func1.regions[0].blocks[0].ops[1]) is Constant
    assert type(func1.regions[0].blocks[0].ops[2]) is Addi
    assert type(func1.regions[0].blocks[1].ops[0]) is Constant
    assert type(func1.regions[0].blocks[1].ops[1]) is Constant
    assert type(func1.regions[0].blocks[1].ops[2]) is Addi
    printer = Printer()
    import pdb;pdb.set_trace()




    def test_memref_load_i32_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    load = Load.get(memref_ssa_value, [index1, index2])

    assert load.memref is memref_ssa_value
    assert load.indices[0] is index1
    assert load.indices[1] is index2
    assert load.res.typ is i32