"""
Test the usage of scf dialect.
"""

from typing import cast

import pytest

from xdsl.builder import Builder
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IndexType, ModuleOp, i32, i64
from xdsl.dialects.scf import For, If, ParallelOp, ReduceOp, ReduceReturnOp, Yield
from xdsl.dialects.test import TestTermOp
from xdsl.ir.core import Block, BlockArgument, Region
from xdsl.utils.exceptions import DiagnosticException, VerifyException


def test_for_with_loop_carried_verify():
    """Test for with loop-carried variables"""

    lower = Constant.from_int_and_width(0, IndexType())
    upper = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())

    @Builder.implicit_region((IndexType(), IndexType()))
    def body(_: tuple[BlockArgument, ...]) -> None:
        Yield.get(carried)

    for_op = For.get(lower, upper, step, [carried], body)

    assert for_op.lb is lower.result
    assert for_op.ub is upper.result
    assert for_op.step is step.result
    assert for_op.iter_args == tuple([carried.result])
    assert for_op.body is body

    assert len(for_op.results) == 1
    assert for_op.results[0].type == carried.result.type
    assert tuple(for_op.operands) == (
        lower.result,
        upper.result,
        step.result,
        carried.result,
    )
    assert for_op.regions == [body]
    assert for_op.attributes == {}

    for_op.verify()


def test_for_without_loop_carried_verify():
    """Test for without loop-carried variables"""

    lower = Constant.from_int_and_width(0, IndexType())
    upper = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())

    @Builder.implicit_region((IndexType(),))
    def body(_: tuple[BlockArgument, ...]) -> None:
        Yield.get()

    for_op = For.get(lower, upper, step, [], body)

    assert for_op.lb is lower.result
    assert for_op.ub is upper.result
    assert for_op.step is step.result
    assert for_op.iter_args == tuple()
    assert for_op.body is body

    assert len(for_op.results) == 0
    assert tuple(for_op.operands) == (
        lower.result,
        upper.result,
        step.result,
    )
    assert for_op.regions == [body]
    assert for_op.attributes == {}

    for_op.verify()


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
    with pytest.raises(DiagnosticException):
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


def test_parallel_verify_only_induction_in_block():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    init_val = Constant.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[IndexType(), i32])
    b.add_op(Yield.get(init_val))

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body, initVals)
    with pytest.raises(VerifyException):
        p.verify()

    b2 = Block(arg_types=[IndexType(), i32, i32])
    b2.add_op(Yield.get(init_val))
    body2 = Region(b2)
    p2 = ParallelOp.get([lbi], [ubi], [si], body2, initVals)
    with pytest.raises(VerifyException):
        p2.verify()


def test_parallel_block_arg_indextype():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    b = Block(arg_types=[IndexType()])
    b.add_op(Yield.get())

    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body)
    p.verify()

    b2 = Block(arg_types=[i32])
    b2.add_op(Yield.get())
    body2 = Region(b2)
    p2 = ParallelOp.get([lbi], [ubi], [si], body2)
    with pytest.raises(VerifyException):
        p2.verify()


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


def test_parallel_verify_yield_zero_ops():
    lbi = Constant.from_int_and_width(0, IndexType())
    ubi = Constant.from_int_and_width(10, IndexType())
    si = Constant.from_int_and_width(1, IndexType())

    val = Constant.from_int_and_width(10, i64)

    b = Block(arg_types=[IndexType()])
    b.add_op(Yield.get(val))
    body = Region(b)
    p = ParallelOp.get([lbi], [ubi], [si], body)
    with pytest.raises(
        VerifyException,
        match="Single-block region terminator scf.yield has 1 operands "
        "but 0 expected inside an scf.parallel",
    ):
        p.verify()

    b2 = Block(arg_types=[IndexType()])
    b2.add_op(Yield.get())
    body2 = Region(b2)
    p2 = ParallelOp.get([lbi], [ubi], [si], body2)
    # Should verify as yield is empty
    p2.verify()


def test_parallel_test_count_number_reduction_ops():
    @Builder.implicit_region
    def body():
        for i in range(10):
            init_val = Constant.from_int_and_width(i, i32)
            ReduceOp.get(init_val, Block())

    p = ParallelOp.get([], [], [], body)
    assert p.count_number_reduction_ops() == 10


def test_parallel_get_arg_type_of_nth_reduction_op():
    @Builder.implicit_region
    def body():
        init_val1 = Constant.from_int_and_width(10, i32)
        init_val2 = Constant.from_int_and_width(10, i64)
        for i in range(10):
            ReduceOp.get(init_val1 if i % 2 == 0 else init_val2, Block())

    p = ParallelOp.get([], [], [], body)
    assert p.count_number_reduction_ops() == 10
    for i in range(10):
        assert p.get_arg_type_of_nth_reduction_op(i) == i32 if i % 2 == 0 else i64


def test_reduce_op():
    init_val = Constant.from_int_and_width(10, i32)

    reduce_op = ReduceOp.get(init_val, Block(arg_types=[i32, i32]))

    assert reduce_op.argument is init_val.results[0]
    assert reduce_op.argument.type is i32
    assert len(reduce_op.body.blocks) == 1
    assert len(reduce_op.body.block.args) == 2
    assert reduce_op.body.block.args[0].type == reduce_op.body.block.args[0].type == i32


def test_reduce_op_num_block_args():
    init_val = Constant.from_int_and_width(10, i32)
    reduce_constant = Constant.from_int_and_width(100, i32)

    with pytest.raises(
        VerifyException,
        match="scf.reduce block must have exactly two arguments, but ",
    ):
        rro = ReduceReturnOp.get(reduce_constant)
        ReduceOp.get(init_val, Block([rro], arg_types=[i32, i32, i32])).verify()

    with pytest.raises(
        VerifyException,
        match="scf.reduce block must have exactly two arguments, but ",
    ):
        rro = ReduceReturnOp.get(reduce_constant)
        ReduceOp.get(init_val, Block([rro], arg_types=[i32])).verify()

    with pytest.raises(
        VerifyException,
        match="scf.reduce block must have exactly two arguments, but ",
    ):
        rro = ReduceReturnOp.get(reduce_constant)
        ReduceOp.get(init_val, Block([rro], arg_types=[])).verify()


def test_reduce_op_num_block_arg_types():
    init_val = Constant.from_int_and_width(10, i32)
    reduce_constant = Constant.from_int_and_width(100, i32)

    with pytest.raises(
        VerifyException,
        match="scf.reduce block argument types must be the same but have",
    ):
        rro = ReduceReturnOp.get(reduce_constant)
        ReduceOp.get(init_val, Block([rro], arg_types=[i32, i64])).verify()

    with pytest.raises(
        VerifyException,
        match="scf.reduce block argument types must be the same but have",
    ):
        rro = ReduceReturnOp.get(reduce_constant)
        ReduceOp.get(init_val, Block([rro], arg_types=[i64, i32])).verify()


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

    with pytest.raises(
        VerifyException,
        match="Block inside scf.reduce must terminate with an scf.reduce.return",
    ):
        ReduceOp.get(
            init_val, Block([TestTermOp.create()], arg_types=[i32, i32])
        ).verify()


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
    assert rro.result.type is i32


def test_reduce_return_type_is_operand_type():
    reduce_constant = Constant.from_int_and_width(100, i32)
    reduce_constant_wrong_type = Constant.from_int_and_width(100, i64)
    rro = ReduceReturnOp.get(reduce_constant_wrong_type)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    init_val = Constant.from_int_and_width(10, i32)
    with pytest.raises(
        VerifyException,
        match="scf.reduce.return result type at end of scf.reduce block must",
    ):
        ReduceOp.get(init_val, reduce_block).verify()


def test_empty_else():
    # create if without an else block:
    m = ModuleOp(
        [
            t := Constant.from_int_and_width(1, 1),
            If.get(
                t,
                [],
                [
                    Yield.get(),
                ],
            ),
        ]
    )

    assert len(cast(If, list(m.ops)[1]).false_region.blocks) == 0
