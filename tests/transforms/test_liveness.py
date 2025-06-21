from xdsl.dialects import test
from xdsl.dialects.builtin import i32
from xdsl.ir import Block, Region
from xdsl.transforms.experimental import liveness


def test_blockinfo_livein():
    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp(operands=[op1.results[0], op2.results[0]], result_types=[i32])
    op4 = test.TestOp(regions=[Region(Block([op3]))])

    # Block mapping
    block_mapping = liveness.build_block_mapping(op4)
    print(block_mapping.keys())
    bm = block_mapping[op4.regions[0].blocks[0]]

    print(bm.in_values)
    print(bm.out_values)

    assert set([op1.results[0], op2.results[0]]) == bm.in_values
    assert set() == bm.out_values

    # Liveness
    _liveness = liveness.Liveness(op4)

    assert set([op1.results[0], op2.results[0]]) == _liveness.get_livein(
        op4.regions[0].blocks[0]
    )


def test_blockinfo_liveout():
    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(operands=[op1.results[0]])
    op3 = test.TestOp(regions=[Region([Block([op1]), Block([op2])])])

    # Block mapping
    block_mapping = liveness.build_block_mapping(op3)
    print(block_mapping.keys())
    bm = block_mapping[op3.regions[0].blocks[0]]

    assert set([op1.results[0]]) == bm.out_values

    # Liveness
    _liveness = liveness.Liveness(op3)

    assert set([op1.results[0]]) == _liveness.get_liveout(op3.regions[0].blocks[0])


def test_dead_value():
    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp(operands=[op1.results[0]])
    op4 = test.TestOp(regions=[Region(Block([op1, op2, op3]))])

    _liveness = liveness.Liveness(op4)
    val = op1.results[0]

    assert not _liveness.is_dead_after(val, op2)
    assert _liveness.is_dead_after(val, op3)


def test_live_operations():
    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp()
    op3 = test.TestOp()
    op4 = test.TestOp(operands=[op1.results[0]])
    op5 = test.TestOp()
    op6 = test.TestOp(regions=[Region([Block([op1, op2, op3]), Block([op4, op5])])])

    _liveness = liveness.Liveness(op6)
    live_ops = _liveness.resolve_liveness(op1.results[0])

    assert op1 in live_ops
    assert op2 in live_ops
    assert op3 in live_ops
    assert op4 in live_ops
    assert op5 not in live_ops
