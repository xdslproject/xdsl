from xdsl.dialects import test
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import MemRefType, i32
from xdsl.dialects.memref import LoadOp, StoreOp
from xdsl.dialects.scf import ForOp
from xdsl.ir import Block
from xdsl.transforms.loop_hoist_memref import (
    find_same_target_store,
    is_loop_dependent,
)
from xdsl.utils.test_value import create_ssa_value


def test_find_same_target_store0():
    i32_memref_type = MemRefType(i32, [1])
    memref_ssa_value = create_ssa_value(i32_memref_type)
    load0 = LoadOp.get(memref_ssa_value, [])

    assert find_same_target_store(load0) is None

    bb0 = Block([load0])

    assert find_same_target_store(load0) is None

    i32_ssa_value = create_ssa_value(i32)
    store0 = StoreOp.get(i32_ssa_value, memref_ssa_value, [])

    assert find_same_target_store(load0) is None

    bb0.add_ops([test.TestOp(), store0])

    assert find_same_target_store(load0) is store0

    store1 = StoreOp.get(i32_ssa_value, memref_ssa_value, [])
    bb0.add_op(store1)

    assert find_same_target_store(load0) is None

    store2 = StoreOp.get(i32_ssa_value, memref_ssa_value, [])
    load1 = LoadOp.get(memref_ssa_value, [])
    Block([store2, load1])

    assert find_same_target_store(load1) is store2


def test_is_loop_dependent_no_dep():
    lb = ConstantOp.from_int_and_width(0, i32)
    ub = ConstantOp.from_int_and_width(42, i32)
    step = ConstantOp.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    bb0 = Block([op1, op2], arg_types=[i32])

    for_op = ForOp(lb, ub, step, [], bb0)

    assert not is_loop_dependent(op1.results[0], for_op)
    assert not is_loop_dependent(op2.results[0], for_op)


def test_is_loop_dependent_no_dep_with_visited():
    lb = ConstantOp.from_int_and_width(0, i32)
    ub = ConstantOp.from_int_and_width(42, i32)
    step = ConstantOp.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp([op2, op1], result_types=[i32])
    op4 = test.TestOp([op3, op2], result_types=[i32])
    bb0 = Block([op1, op2, op3, op4], arg_types=[i32])

    for_op = ForOp(lb, ub, step, [], bb0)

    assert not is_loop_dependent(op1.results[0], for_op)
    assert not is_loop_dependent(op2.results[0], for_op)
    assert not is_loop_dependent(op3.results[0], for_op)
    assert not is_loop_dependent(op4.results[0], for_op)


def test_is_loop_dependent_dep():
    lb = ConstantOp.from_int_and_width(0, i32)
    ub = ConstantOp.from_int_and_width(42, i32)
    step = ConstantOp.from_int_and_width(3, i32)

    bb0 = Block(arg_types=[i32])
    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp([op2, op1], result_types=[i32])
    op4 = test.TestOp([bb0.args[0], op2], result_types=[i32])
    bb0.add_ops([op1, op2, op3, op4])

    for_op = ForOp(lb, ub, step, [], bb0)

    assert not is_loop_dependent(op1.results[0], for_op)
    assert not is_loop_dependent(op2.results[0], for_op)
    assert not is_loop_dependent(op3.results[0], for_op)
    assert is_loop_dependent(op4.results[0], for_op)
