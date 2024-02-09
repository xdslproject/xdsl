import pytest

from xdsl.dialects import test, scf, builtin, memref
from xdsl.ir import Block

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
    i32_memref_type = builtin.MemRefType(builtin.i32, [1])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    load0 = memref.Load.get(memref_ssa_value, [])

    assert find_same_target_store(load0) is None

    bb0 = Block([load0])

    assert find_same_target_store(load0) is None

    i32_ssa_value = TestSSAValue(builtin.i32)
    store0 = memref.Store.get(i32_ssa_value, memref_ssa_value, [])

    assert find_same_target_store(load0) is None

    bb0.add_ops([test.TestOp(), store0])

    assert find_same_target_store(load0) is store0

    store1 = memref.Store.get(i32_ssa_value, memref_ssa_value, [])
    bb0.add_op(store1)

    assert find_same_target_store(load0) is None

    store2 = memref.Store.get(i32_ssa_value, memref_ssa_value, [])
    load1 = memref.Load.get(memref_ssa_value, [])
    Block([store2, load1])

    assert find_same_target_store(load1) is store2
