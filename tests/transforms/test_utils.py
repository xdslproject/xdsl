import pytest

from xdsl.dialects import test, arith
from xdsl.ir import Block
from xdsl.dialects.scf import For
from xdsl.dialects.builtin import i32, MemRefType
from xdsl.dialects.memref import Load, Store
from xdsl.dialects.arith import Constant

from xdsl.transforms.utils import (
    find_same_target_store,
    get_operation_at_index,
    is_loop_dependent,
)
from xdsl.utils.test_value import TestSSAValue


def test_get_operation_at_index():
    bb0 = Block()

    with pytest.raises(ValueError):
        get_operation_at_index(bb0, 0)

    with pytest.raises(ValueError):
        get_operation_at_index(bb0, 1)

    with pytest.raises(ValueError):
        get_operation_at_index(bb0, -1)

    op0 = test.TestOp()
    bb1 = Block([op0])

    assert op0 == get_operation_at_index(bb1, 0)

    with pytest.raises(ValueError):
        get_operation_at_index(bb1, 1)

    with pytest.raises(ValueError):
        get_operation_at_index(bb1, -1)

    op1 = test.TestOp()
    op2 = test.TestOp()
    bb2 = Block([op1, op2])

    assert op1 == get_operation_at_index(bb2, 0)
    assert op2 == get_operation_at_index(bb2, 1)

    with pytest.raises(ValueError):
        get_operation_at_index(bb2, 2)


def test_find_same_target_store0():
    i32_memref_type = MemRefType(i32, [1])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    load0 = Load.get(memref_ssa_value, [])

    assert find_same_target_store(load0) is None

    bb0 = Block([load0])

    assert find_same_target_store(load0) is None

    i32_ssa_value = TestSSAValue(i32)
    store0 = Store.get(i32_ssa_value, memref_ssa_value, [])

    assert find_same_target_store(load0) is None

    bb0.add_ops([test.TestOp(), store0])

    assert find_same_target_store(load0) is store0

    store1 = Store.get(i32_ssa_value, memref_ssa_value, [])
    bb0.add_op(store1)

    assert find_same_target_store(load0) is None

    store2 = Store.get(i32_ssa_value, memref_ssa_value, [])
    load1 = Load.get(memref_ssa_value, [])
    Block([store2, load1])

    assert find_same_target_store(load1) is store2


def test_is_loop_dependent_no_dep():
    lb = Constant.from_int_and_width(0, i32)
    ub = Constant.from_int_and_width(42, i32)
    step = Constant.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    bb0 = Block([op1, op2], arg_types=[i32])

    for_op = For(lb, ub, step, [], bb0)

    assert not is_loop_dependent(*op1.res, for_op)
    assert not is_loop_dependent(*op2.res, for_op)


def test_is_loop_dependent_no_dep_with_visited():
    lb = Constant.from_int_and_width(0, i32)
    ub = Constant.from_int_and_width(42, i32)
    step = Constant.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp([op1, op2], result_types=[i32])
    op4 = test.TestOp([op1, op2], result_types=[i32])
    bb0 = Block([op1, op2, op3, op4], arg_types=[i32])

    for_op = For(lb, ub, step, [], bb0)

    assert not is_loop_dependent(*op1.res, for_op)
    assert not is_loop_dependent(*op2.res, for_op)
    assert not is_loop_dependent(*op3.res, for_op)
    assert not is_loop_dependent(*op4.res, for_op)
