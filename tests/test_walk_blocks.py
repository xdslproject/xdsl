from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Block, Region


def test_preorder_walk():
    ctx = Context()
    ctx.load_dialect(test.Test)

    # Create a region with multiple blocks
    outer_region = Region([Block(), Block()])
    outer_block1, outer_block2 = outer_region.blocks

    # Create nested regions with multiple blocks
    nested_region1 = Region([Block(), Block()])
    nested_block1, nested_block2 = nested_region1.blocks

    nested_region2 = Region([Block(), Block()])
    nested_block3, nested_block4 = nested_region2.blocks

    # Create a third region for multi-region op
    nested_region3 = Region([Block(), Block()])
    nested_block5, nested_block6 = nested_region3.blocks

    # Build operations with nested regions
    root_op = test.TestOp(regions=(outer_region,))

    with ImplicitBuilder(outer_block1):
        test.TestOp(attributes={"id": IntAttr(1)})
        with ImplicitBuilder(nested_block1):
            test.TestOp(attributes={"id": IntAttr(3)})
            test.TestTermOp()
        with ImplicitBuilder(nested_block2):
            test.TestOp(attributes={"id": IntAttr(4)})
            test.TestTermOp()
        test.TestOp(regions=(nested_region1,))
        test.TestTermOp()

    with ImplicitBuilder(outer_block2):
        test.TestOp(attributes={"id": IntAttr(2)})
        with ImplicitBuilder(nested_block3):
            test.TestOp(attributes={"id": IntAttr(5)})
            test.TestTermOp()
        with ImplicitBuilder(nested_block4):
            test.TestOp(attributes={"id": IntAttr(6)})
            test.TestTermOp()
        # Add multi-region op
        with ImplicitBuilder(nested_block5):
            test.TestOp(attributes={"id": IntAttr(7)})
            test.TestTermOp()
        with ImplicitBuilder(nested_block6):
            test.TestOp(attributes={"id": IntAttr(8)})
            test.TestTermOp()
        test.TestOp(regions=(nested_region2, nested_region3))
        test.TestTermOp()

    assert tuple(root_op.walk_blocks()) == (
        outer_block1,
        nested_block1,
        nested_block2,
        outer_block2,
        nested_block3,
        nested_block4,
        nested_block5,
        nested_block6,
    )

    assert tuple(outer_block1.walk_blocks()) == (
        outer_block1,
        nested_block1,
        nested_block2,
    )

    assert tuple(outer_block2.walk_blocks()) == (
        outer_block2,
        nested_block3,
        nested_block4,
        nested_block5,
        nested_block6,
    )

    assert tuple(root_op.walk_blocks(reverse=True)) == (
        nested_block6,
        nested_block5,
        nested_block4,
        nested_block3,
        outer_block2,
        nested_block2,
        nested_block1,
        outer_block1,
    )

    assert tuple(outer_block1.walk_blocks(reverse=True)) == (
        nested_block2,
        nested_block1,
        outer_block1,
    )

    assert tuple(outer_block2.walk_blocks(reverse=True)) == (
        nested_block6,
        nested_block5,
        nested_block4,
        nested_block3,
        outer_block2,
    )
