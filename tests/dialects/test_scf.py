import pytest
from typing import cast
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import Region, IndexType, ModuleOp, i32, i64
from xdsl.dialects.cf import Block
from xdsl.dialects.scf import For, ParallelOp, If, Yield, ReduceOp, ReduceReturnOp
from xdsl.utils.exceptions import VerifyException


def test_for():
    lb = Constant.from_int_and_width(0, IndexType())
    ub = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())
    bodyblock = Block(arg_types=[IndexType()])
    body = Region(bodyblock)
    f = For.get(lb, ub, step, [carried], body)

    assert f.lb is lb.result
    assert f.ub is ub.result
    assert f.step is step.result
    assert f.iter_args == tuple([carried.result])
    assert f.body is body

    assert len(f.results) == 1
    assert f.results[0].typ == carried.result.typ
    assert f.operands == (lb.result, ub.result, step.result, carried.result)
    assert f.regions == [body]
    assert f.attributes == {}


def test_parallel_no_init_vals():
    lbi = Constant.from_int_and_width(0, IndexType())
    lbj = Constant.from_int_and_width(1, IndexType())
    lbk = Constant.from_int_and_width(18, IndexType())

    ubi = Constant.from_int_and_width(10, IndexType())
    ubj = Constant.from_int_and_width(110, IndexType())
    ubk = Constant.from_int_and_width(92, IndexType())

    si = Constant.from_int_and_width(1, IndexType())
    sj = Constant.from_int_and_width(3, IndexType())
    sk = Constant.from_int_and_width(8, IndexType())

    body = Region()

    lowerBounds = [lbi, lbj, lbk]
    upperBounds = [ubi, ubj, ubk]
    steps = [si, sj, sk]

    p = ParallelOp.get(lowerBounds, upperBounds, steps, body)

    assert isinstance(p, ParallelOp)
    assert p.lowerBound == tuple(l.result for l in lowerBounds)
    assert p.upperBound == tuple(l.result for l in upperBounds)
    assert p.step == tuple(l.result for l in steps)
    assert p.body is body


def test_parallel_with_init_vals():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    body = Region()

    init_val = Constant.from_int_and_width(10, i32)

    initVals = [init_val]

    lowerBounds = [lbi]
    upperBounds = [ubi]
    steps = [si]

    p = ParallelOp.get(lowerBounds, upperBounds, steps, body, initVals)

    assert isinstance(p, ParallelOp)
    assert p.lowerBound == tuple(l.result for l in lowerBounds)
    assert p.upperBound == tuple(l.result for l in upperBounds)
    assert p.step == tuple(l.result for l in steps)
    assert p.body is body
    assert p.initVals == tuple(l.result for l in initVals)


def test_parallel_verify_one_block():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    body = Region()
    p = ParallelOp.get([lbi], [ubi], [si], body)
    with pytest.raises(VerifyException):
        p.verify()


def test_parallel_verify_num_bounds_equal():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    lowerBounds = [lbi, lbi]
    upperBounds = [ubi]
    steps = [si]

    body = Region(Block())

    p = ParallelOp.get(lowerBounds, upperBounds, steps, body)
    with pytest.raises(VerifyException):
        p.verify()

    upperBounds = [ubi, ubi]

    body2 = Region(Block())
    p2 = ParallelOp.get(lowerBounds, upperBounds, steps, body2)
    with pytest.raises(VerifyException):
        p2.verify()

    lowerBounds = [lbi]
    upperBounds = [ubi]
    steps = [si, si]

    body3 = Region(Block())
    p3 = ParallelOp.get(lowerBounds, upperBounds, steps, body3)
    with pytest.raises(VerifyException):
        p3.verify()


def test_parallel_verify_num_args_correct():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[IndexType(), i32])
    b.add_op(Yield.get(init_val))

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, initVals)
    # This should verify
    p.verify()

    b2 = Block(arg_types=[IndexType(), i32, i32])
    b2.add_op(Yield.get(init_val))
    body2 = Region(b2)
    p2 = ParallelOp.get([lbi], [ubi], [si], body2, initVals)
    with pytest.raises(VerifyException):
        p2.verify()


def test_parallel_verify_omitted_induction_var_in_block():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[i32])
    b.add_op(Yield.get(init_val))

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, initVals)
    with pytest.raises(VerifyException):
        p.verify()


def test_parallel_verify_initVar_and_block_type():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[IndexType(), i64])
    b.add_op(Yield.get(init_val))

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, initVals)
    with pytest.raises(VerifyException):
        p.verify()


def test_parallel_verify_reduction_and_block_type():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[IndexType()])

    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    b.add_op(ReduceOp.get(init_val, reduce_block))
    b.add_op(Yield.get())

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, initVals)
    # This should verify
    p.verify()


def test_parallel_verify_reduction_and_block_type_fails():
    # This also covers the return type test too, as the return type
    # is driven by the init_val
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i64)

    initVals = [init_val]

    b = Block(arg_types=[IndexType()])

    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    b.add_op(ReduceOp.get(init_val, reduce_block))
    b.add_op(Yield.get())

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, initVals)
    with pytest.raises(VerifyException):
        p.verify()


def test_parallel_verify_yield_last_op():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    b = Block(arg_types=[IndexType()])
    b.add_op(Yield.get())

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body)
    # This should verify
    p.verify()

    p2 = ParallelOp.get([lbi], [ubi], [si],
                        Region(Block(arg_types=[IndexType()])))
    with pytest.raises(VerifyException):
        p2.verify()

    b3 = Block(arg_types=[IndexType()])
    b3.add_op(Constant.from_int_and_width(1, IndexType()))
    p3 = ParallelOp.get([lbi], [ubi], [si], Region(b3))
    with pytest.raises(VerifyException):
        p3.verify()


def test_parallel_verify_yield_num_ops():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    val = Constant.from_int_and_width(10, i64)

    b = Block(arg_types=[IndexType()])
    b.add_op(Yield.get(val))
    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body)
    with pytest.raises(VerifyException):
        p.verify()

    init_val = Constant.from_int_and_width(10, i64)
    b2 = Block(arg_types=[IndexType(), i64])
    b2.add_op(Yield.get())
    body2 = Region(b2)
    p2 = ParallelOp.get([lbi], [ubi], [si], body2, [init_val])
    with pytest.raises(VerifyException):
        p2.verify()


def test_parallel_verify_yield_types_match_result_types():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i32)

    b = Block(arg_types=[IndexType(), i32])
    ret_val = Constant.from_int_and_width(10, i64)
    b.add_ops([ret_val, Yield.get(ret_val)])

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, [init_val])
    with pytest.raises(VerifyException):
        p.verify()


def test_parallel_test_count_number_reduction_ops():
    init_val = Constant.from_int_and_width(10, i32)
    reductions = []
    for i in range(10):
        reductions.append(ReduceOp.get(init_val, Block()))

    b = Block()
    b.add_ops(reductions)
    p = ParallelOp.get([], [], [], Region(b))
    assert p.count_number_reduction_ops() == 10


def test_parallel_get_arg_type_of_nth_reduction_op():
    init_val1 = Constant.from_int_and_width(10, i32)
    init_val2 = Constant.from_int_and_width(10, i64)
    reductions = []
    for i in range(10):
        reductions.append(
            ReduceOp.get(init_val1 if i % 2 == 0 else init_val2, Block()))

    b = Block()
    b.add_ops(reductions)
    p = ParallelOp.get([], [], [], Region(b))
    assert p.count_number_reduction_ops() == 10
    for i in range(10):
        assert p.get_arg_type_of_nth_reduction_op(
            i) == i32 if i % 2 == 0 else i64


def test_reduce_op():
    init_val = Constant.from_int_and_width(10, i32)

    reduce_op = ReduceOp.get(init_val, Block(arg_types=[i32, i32]))

    assert reduce_op.argument is init_val.results[0]
    assert reduce_op.argument.typ is i32
    assert len(reduce_op.body.blocks) == 1
    assert len(reduce_op.body.block.args) == 2
    assert reduce_op.body.block.args[0].typ == reduce_op.body.block.args[
        0].typ == i32


def test_reduce_op_num_block_args():
    init_val = Constant.from_int_and_width(10, i32)

    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[i32, i32, i32])).verify()
    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[i32])).verify()
    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[])).verify()


def test_reduce_op_num_block_arg_types():
    init_val = Constant.from_int_and_width(10, i32)

    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[i32, i64])).verify()

    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[i64, i32])).verify()


def test_reduce_op_num_block_arg_types_match_operand_type():
    init_val = Constant.from_int_and_width(10, i32)

    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[i64, i64])).verify()


def test_reduce_return_op_at_end():
    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    init_val = Constant.from_int_and_width(10, i32)
    ReduceOp.get(init_val, reduce_block).verify()

    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, Block(arg_types=[i32, i32])).verify()

    reduce_constant.detach()
    reduce_block2 = Block(arg_types=[i32, i32])
    reduce_block2.add_ops([reduce_constant])
    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, reduce_block2).verify()


def test_reduce_return_type_is_arg_type():
    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    init_val = Constant.from_int_and_width(10, i64)
    with pytest.raises(VerifyException):
        ReduceOp.get(init_val, reduce_block).verify()


def test_reduce_return_op():
    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)

    assert rro.result is reduce_constant.results[0]
    assert rro.result.typ is i32


def test_reduce_return_op_not_part_of_reduce():
    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)
    with pytest.raises(Exception):
        rro.verify()
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])
    with pytest.raises(Exception):
        rro.verify()
    region = Region(reduce_block)
    with pytest.raises(Exception):
        rro.verify()

    region.detach_block(reduce_block)
    init_val = Constant.from_int_and_width(10, i32)
    ReduceOp.get(init_val, reduce_block)
    rro.verify()


def test_reduce_return_op_not_at_end():
    reduce_constant = Constant.from_int_and_width(100, i32)
    rro = ReduceReturnOp.get(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([rro, reduce_constant])
    init_val = Constant.from_int_and_width(10, i32)
    ReduceOp.get(init_val, reduce_block)
    with pytest.raises(VerifyException):
        rro.verify()


def test_empty_else():
    # create if without an else block:
    m = ModuleOp([
        t := Constant.from_int_and_width(1, 1),
        If.get(t, [], [
            Yield.get(),
        ]),
    ])

    assert len(cast(If, m.ops[1]).false_region.blocks) == 0
