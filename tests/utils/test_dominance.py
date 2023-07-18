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
from xdsl.utils.dominance import dominance_tree


@irdl_op_definition
class VarSuccessorOp(IRDLOperation):
    """Utility operation that requires successors."""

    name = "test.successor_op"

    successor: VarSuccessor = var_successor_def()
    res: OptOpResult = opt_result_def()

    traits = frozenset([IsTerminator()])


def test_domimance_tree_base_cases():
    """Tests dominance of an SSACFG region."""

    # test empty region
    assert dominance_tree(Region()) == {}

    # test single-block region with empty block
    block0 = Block()
    assert dominance_tree(Region(block0)) == {block0: None}

    # test multi-block region with empty blocks
    blocks = [Block(), Block()]
    assert dominance_tree(Region(blocks)) == {blocks[0]: None}

    # test single-block region with no successors
    block0 = Block([test.TestTermOp.create()])
    region0 = Region([block0])
    region0.verify()
    assert dominance_tree(region0) == {block0: None}

    # test multi-block region with no successors
    block0 = Block([test.TestOp.create(), test.TestTermOp.create()])
    region0 = Region([block0, Block()])
    region0.verify()
    assert dominance_tree(region0) == {block0: None}

    # test single-block region with successor cycle
    block0 = Block()
    op0 = VarSuccessorOp.create(successors=[block0])
    block0.add_op(op0)
    region0 = Region([block0])
    region0.verify()
    assert dominance_tree(region0) == {block0: None}

    # test multi-block region with redundant same successor
    block1 = Block()
    block0 = Block([VarSuccessorOp.create(successors=[block1, block1])])
    region0 = Region([block0, block1])
    region0.verify()
    assert dominance_tree(region0) == {block0: None, block1: block0}

    # test multi-block region with single successor cycle
    block0, block1 = Block(), Block()
    op0 = VarSuccessorOp.create(successors=[block1])
    op1 = VarSuccessorOp.create(successors=[block0])
    block0.add_op(op0)
    block1.add_op(op1)
    region0 = Region([block0, block1])
    region0.verify()
    assert dominance_tree(region0) == {block0: None, block1: block0}


def test_domimance_tree_case1():
    # test multi-block region with successors: case 1
    block0, block1, block2, block3 = [Block() for _ in range(4)]

    ops0 = [
        test.TestOp.create(),
        test.TestOp.create(),
        VarSuccessorOp.create(successors=[block1, block2]),
    ]
    block0.add_ops(ops0)

    ops1 = [test.TestOp.create(), VarSuccessorOp.create(successors=[block3])]
    block1.add_ops(ops1)

    op2 = VarSuccessorOp.create(successors=[block3])
    block2.add_op(op2)

    region0 = Region([block0, block1, block2, block3])
    region0.verify()

    from xdsl.printer import Printer

    pp = Printer()
    pp.print_region(region0)

    assert dominance_tree(region0) == {
        block0: None,
        block1: block0,
        block2: block0,
        block3: block0,
    }


def test_domimance_tree_case2():
    # test multi-block region with successors: case 2
    block0, block1, block2, block3, block4 = [Block() for _ in range(5)]

    ops0 = [
        test.TestOp.create(),
        test.TestOp.create(),
        VarSuccessorOp.create(successors=[block1, block2]),
    ]
    block0.add_ops(ops0)

    ops1 = [test.TestOp.create(), VarSuccessorOp.create(successors=[block3])]
    block1.add_ops(ops1)

    op2 = VarSuccessorOp.create(successors=[block4])
    block2.add_op(op2)

    region0 = Region([block0, block1, block2, block3, block4])
    region0.verify()

    assert dominance_tree(region0) == {
        block0: None,
        block1: block0,
        block2: block0,
        block3: block1,
        block4: block2,
    }


def test_domimance_tree_case3():
    block0, block1, block2, block3, block4, block5 = [Block() for _ in range(6)]

    ops0 = [
        test.TestOp.create(),
        test.TestOp.create(),
        VarSuccessorOp.create(successors=[block1, block2]),
    ]
    block0.add_ops(ops0)

    op2 = VarSuccessorOp.create(successors=[block3, block5])
    block2.add_op(op2)

    op3 = VarSuccessorOp.create(successors=[block2, block4, block5])
    block3.add_op(op3)

    region0 = Region([block0, block1, block2, block3, block4, block5])
    region0.verify()

    assert dominance_tree(region0) == {
        block0: None,
        block1: block0,
        block2: block0,
        block3: block2,
        block4: block3,
        block5: block2,
    }
