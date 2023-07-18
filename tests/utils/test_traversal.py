from xdsl.dialects import test
from xdsl.ir import Block, Region
from xdsl.irdl import (
    IRDLOperation,
    OptOpResult,
    VarSuccessor,
    irdl_op_definition,
    opt_result_def,
    var_successor_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.traversal import postorder, predecessors


@irdl_op_definition
class VarSuccessorOp(IRDLOperation):
    """Utility operation that requires successors."""

    name = "test.successor_op"

    successor: VarSuccessor = var_successor_def()
    res: OptOpResult = opt_result_def()

    traits = frozenset([IsTerminator()])


def test_predecessors():
    """Tests predecessors of an SSACFG region."""

    # test empty region
    assert predecessors(Region()) == {}

    # test single-block region with empty block
    assert predecessors(Region(Block())) == {}

    # test multi-block region with empty blocks
    assert predecessors(Region([Block(), Block()])) == {}

    # test single-block region with no successors
    block0 = Block([test.TestTermOp.create()])
    region0 = Region([block0])
    region0.verify()
    assert predecessors(region0) == {}

    # test multi-block region with no successors
    block0 = Block([test.TestOp.create(), test.TestTermOp.create()])
    region0 = Region([block0, Block()])
    region0.verify()
    assert predecessors(region0) == {}

    # test single-block region with successor cycle
    block0 = Block()
    op0 = VarSuccessorOp.create(successors=[block0])
    block0.add_op(op0)
    region0 = Region([block0])
    region0.verify()
    assert predecessors(region0) == {block0: {block0}}

    # test multi-block region with single successor
    block1 = Block()
    block0 = Block([VarSuccessorOp.create(successors=[block1])])
    region0 = Region([block0, block1])
    region0.verify()
    assert predecessors(region0) == {block1: {block0}}

    # test multi-block region with redundant same successor
    block1 = Block()
    block0 = Block([VarSuccessorOp.create(successors=[block1, block1])])
    region0 = Region([block0, block1])
    region0.verify()
    assert predecessors(region0) == {block1: {block0}}

    # test multi-block region with single successor cycle
    block0, block1 = Block(), Block()
    op0 = VarSuccessorOp.create(successors=[block1])
    op1 = VarSuccessorOp.create(successors=[block0])
    block0.add_op(op0)
    block1.add_op(op1)
    region0 = Region([block0, block1])
    region0.verify()
    assert predecessors(region0) == {
        block0: {block1},
        block1: {block0},
    }

    # test multi-block region with successors
    block0, block1, block2 = Block(), Block(), Block()

    op0 = VarSuccessorOp.create(successors=[block1, block2])
    block0 = Block([op0])

    ops1 = [test.TestOp.create(), test.TestTermOp.create()]
    block1.add_ops(ops1)

    op2 = VarSuccessorOp.create(successors=[block1])
    block2.add_op(op2)

    region0 = Region([block0, block1, block2])
    region0.verify()

    assert predecessors(region0) == {
        block1: {block0, block2},
        block2: {block0},
    }


def test_postorder():
    """Tests the postorder traversal of an SSACFG region."""

    # test empty region
    assert postorder(Region()) == []

    # test single-block region with empty block
    block0 = Block()
    assert postorder(Region(block0)) == [block0]

    # test multi-block region with empty blocks
    blocks = [Block(), Block()]
    assert postorder(Region(blocks)) == [blocks[0]]

    # test single-block region with no successors
    block0 = Block([test.TestTermOp.create()])
    region0 = Region([block0])
    region0.verify()
    assert postorder(region0) == [block0]

    # test multi-block region with no successors
    block0 = Block([test.TestOp.create(), test.TestTermOp.create()])
    region0 = Region([block0, Block()])
    region0.verify()
    assert postorder(region0) == [block0]

    # test single-block region with successor cycle
    block0 = Block()
    op0 = VarSuccessorOp.create(successors=[block0])
    block0.add_op(op0)
    region0 = Region([block0])
    region0.verify()
    assert postorder(region0) == [block0]

    # test multi-block region with single successor
    block1 = Block()
    block0 = Block([VarSuccessorOp.create(successors=[block1])])
    region0 = Region([block0, block1])
    region0.verify()
    assert postorder(region0) == [block1, block0]

    # test multi-block region with redundant same successor
    block1 = Block()
    block0 = Block([VarSuccessorOp.create(successors=[block1, block1])])
    region0 = Region([block0, block1])
    region0.verify()
    assert postorder(region0) == [block1, block0]

    # test multi-block region with single successor cycle
    block0, block1 = Block(), Block()
    op0 = VarSuccessorOp.create(successors=[block1])
    op1 = VarSuccessorOp.create(successors=[block0])
    block0.add_op(op0)
    block1.add_op(op1)
    region0 = Region([block0, block1])
    region0.verify()
    assert postorder(region0) == [block1, block0]

    # test multi-block region with successors
    block0, block1, block2 = Block(), Block(), Block()

    op0 = VarSuccessorOp.create(successors=[block1, block2])
    block0 = Block([op0])

    ops1 = [test.TestOp.create(), test.TestTermOp.create()]
    block1.add_ops(ops1)

    op2 = VarSuccessorOp.create(successors=[block1])
    block2.add_op(op2)

    region0 = Region([block0, block1, block2])
    region0.verify()

    assert postorder(region0) == [block1, block2, block0]
