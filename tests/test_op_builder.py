import pytest

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntAttr, IntegerAttr, i32, i64
from xdsl.dialects.scf import IfOp
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, BlockArgument, Operation, Region
from xdsl.rewriter import BlockInsertPoint, InsertPoint


def test_insertion_point_constructors():
    target = Block(
        [
            (op1 := ConstantOp.from_int_and_width(0, 1)),
            (op2 := ConstantOp.from_int_and_width(1, 1)),
        ]
    )

    assert InsertPoint.at_start(target) == InsertPoint(target, op1)
    assert InsertPoint.at_end(target) == InsertPoint(target, None)
    assert InsertPoint.before(op1) == InsertPoint(target, op1)
    assert InsertPoint.after(op1) == InsertPoint(target, op2)
    assert InsertPoint.before(op2) == InsertPoint(target, op2)
    assert InsertPoint.after(op2) == InsertPoint(target, None)


def test_block_insertion_point_constructors():
    target = Region(
        [
            (block1 := Block()),
            (block2 := Block()),
        ]
    )

    assert BlockInsertPoint.at_start(target) == BlockInsertPoint(target, block1)
    assert BlockInsertPoint.at_end(target) == BlockInsertPoint(target, None)
    assert BlockInsertPoint.before(block1) == BlockInsertPoint(target, block1)
    assert BlockInsertPoint.after(block1) == BlockInsertPoint(target, block2)
    assert BlockInsertPoint.before(block2) == BlockInsertPoint(target, block2)
    assert BlockInsertPoint.after(block2) == BlockInsertPoint(target, None)

    assert BlockInsertPoint.at_start(target) != BlockInsertPoint.at_end(target)


def test_block_insertion_init_incorrect():
    region = Region()
    block = Block()
    with pytest.raises(
        ValueError, match="Insertion point must be in the builder's `region`"
    ):
        BlockInsertPoint(region, block)


def test_block_insertion_point_orphan():
    block = Block()
    with pytest.raises(
        ValueError, match="Block insertion point must have a parent region"
    ):
        BlockInsertPoint.before(block)

    with pytest.raises(
        ValueError, match="Block insertion point must have a parent region"
    ):
        BlockInsertPoint.after(block)


def test_builder():
    target = Block(
        [
            ConstantOp.from_int_and_width(0, 1),
            ConstantOp.from_int_and_width(1, 1),
        ]
    )

    block = Block()
    b = Builder(InsertPoint.at_end(block))

    x = ConstantOp.from_int_and_width(0, 1)
    y = ConstantOp.from_int_and_width(1, 1)

    b.insert(x)
    b.insert(y)

    assert target.is_structurally_equivalent(block)


def test_builder_insertion_point():
    target = Block(
        [
            ConstantOp.from_int_and_width(1, 8),
            ConstantOp.from_int_and_width(2, 8),
            ConstantOp.from_int_and_width(3, 8),
        ]
    )

    block = Block()
    b = Builder(InsertPoint.at_end(block))

    x = ConstantOp.from_int_and_width(1, 8)
    y = ConstantOp.from_int_and_width(2, 8)
    z = ConstantOp.from_int_and_width(3, 8)

    b.insert(x)
    b.insert(z)

    b.insertion_point = InsertPoint.before(z)

    b.insert(y)

    assert target.is_structurally_equivalent(block)


def test_builder_create_block():
    block1 = Block()
    block2 = Block()
    target = Region([block1, block2])
    builder = Builder(InsertPoint.at_start(block1))

    new_block1 = builder.create_block(BlockInsertPoint.at_start(target), (i32,))
    assert len(new_block1.args) == 1
    assert new_block1.args[0].type == i32
    assert len(target.blocks) == 3
    assert target.blocks[0] == new_block1
    assert builder.insertion_point == InsertPoint.at_start(new_block1)

    new_block2 = builder.create_block(BlockInsertPoint.at_end(target), (i64,))
    assert len(new_block2.args) == 1
    assert new_block2.args[0].type == i64
    assert len(target.blocks) == 4
    assert target.blocks[3] == new_block2
    assert builder.insertion_point == InsertPoint.at_start(new_block2)

    new_block3 = builder.create_block(BlockInsertPoint.before(block2), (i32, i64))
    assert len(new_block3.args) == 2
    assert new_block3.args[0].type == i32
    assert new_block3.args[1].type == i64
    assert len(target.blocks) == 5
    assert target.blocks[2] == new_block3
    assert builder.insertion_point == InsertPoint.at_start(new_block3)

    new_block4 = builder.create_block(BlockInsertPoint.after(block2), (i64, i32))
    assert len(new_block4.args) == 2
    assert new_block4.args[0].type == i64
    assert new_block4.args[1].type == i32
    assert len(target.blocks) == 6
    assert target.blocks[4] == new_block4
    assert builder.insertion_point == InsertPoint.at_start(new_block4)


def test_builder_listener_op_insert():
    block = Block()
    b = Builder(InsertPoint.at_end(block))

    x = ConstantOp.from_int_and_width(1, 32)
    y = ConstantOp.from_int_and_width(2, 32)
    z = ConstantOp.from_int_and_width(3, 32)

    added_ops: list[Operation] = []

    def add_op_on_insert(op: Operation):
        added_ops.append(op)

    b.operation_insertion_handler = [add_op_on_insert]

    b.insert(x)
    b.insert(z)
    b.insertion_point = InsertPoint.before(z)
    b.insert(y)

    assert added_ops == [x, z, y]


def test_builder_listener_block_created():
    block = Block()
    region = Region([block])
    b = Builder(InsertPoint.at_start(block))

    created_blocks: list[Block] = []

    def add_block_on_create(b: Block):
        created_blocks.append(b)

    b.block_creation_handler = [add_block_on_create]

    b1 = b.create_block(BlockInsertPoint.at_start(region))
    b2 = b.create_block(BlockInsertPoint.at_end(region))
    b3 = b.create_block(BlockInsertPoint.before(block))
    b4 = b.create_block(BlockInsertPoint.after(block))

    assert created_blocks == [b1, b2, b3, b4]


def test_builder_name_hint_listener():
    block = Block()
    b = Builder(InsertPoint.at_start(block))
    assert b.insert_op(TestOp((), result_types=(i32,))).results[0].name_hint is None

    b.name_hint = "hello"
    # No name hint
    assert b.insert_op(TestOp((), result_types=(i32,))).results[0].name_hint == "hello"

    # With name hint
    op = TestOp((), result_types=(i32,))
    op.results[0].name_hint = "world"
    assert b.insert_op(op).results[0].name_hint == "world"

    with pytest.raises(ValueError, match="Invalid SSAValue name format `1`."):
        b.name_hint = "1"


def test_build_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                ConstantOp.from_int_and_width(one, i32),
                ConstantOp.from_int_and_width(two, i32),
            ]
        )
    )

    @Builder.region
    def region(b: Builder):
        x = ConstantOp.from_int_and_width(one, i32)
        y = ConstantOp.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    assert target.is_structurally_equivalent(region)


def test_build_callable_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                ConstantOp.from_int_and_width(one, i32),
                ConstantOp.from_int_and_width(two, i32),
            ],
            arg_types=(i32,),
        )
    )

    @Builder.region([i32])
    def region(b: Builder, args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        x = ConstantOp.from_int_and_width(one, i32)
        y = ConstantOp.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    assert target.is_structurally_equivalent(region)


def test_build_implicit_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                ConstantOp.from_int_and_width(one, i32),
                ConstantOp.from_int_and_width(two, i32),
            ]
        )
    )

    @Builder.implicit_region
    def region():
        ConstantOp.from_int_and_width(one, i32)
        ConstantOp.from_int_and_width(two, i32)

    assert target.is_structurally_equivalent(region)


def test_build_implicit_callable_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                ConstantOp.from_int_and_width(one, i32),
                ConstantOp.from_int_and_width(two, i32),
            ],
            arg_types=(i32,),
        )
    )

    @Builder.implicit_region([i32])
    def region(args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        ConstantOp.from_int_and_width(one, i32)
        ConstantOp.from_int_and_width(two, i32)

    assert target.is_structurally_equivalent(region)


def test_build_nested_implicit_region():
    target = Region(
        Block(
            [
                cond := ConstantOp.from_int_and_width(1, 1),
                IfOp(
                    cond,
                    (),
                    Region(
                        Block(
                            [
                                ConstantOp.from_int_and_width(2, i32),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    @Builder.implicit_region
    def region():
        cond = ConstantOp.from_int_and_width(1, 1).result

        @Builder.implicit_region
        def then():
            _y = ConstantOp.from_int_and_width(2, i32)

        IfOp(cond, (), then)

    assert target.is_structurally_equivalent(region)


def test_build_implicit_region_fail():
    with pytest.raises(
        ValueError,
        match="Cannot insert operation explicitly when an implicit builder exists.",
    ):
        one = IntAttr(1)
        two = IntAttr(2)
        three = IntAttr(3)

        @Builder.implicit_region
        def region():
            cond = ConstantOp.from_int_and_width(1, 1).result

            _x = ConstantOp.from_int_and_width(one, i32)

            @Builder.implicit_region
            def then_0():
                _y = ConstantOp.from_int_and_width(two, i32)

                @Builder.region
                def then_1(b: Builder):
                    b.insert(ConstantOp.from_int_and_width(three, i32))

                IfOp(cond, (), then_1)

            IfOp(cond, (), then_0)

        _ = region


def test_try_fold_foldable_operation():
    """Test that try_fold correctly folds an AddiOp with zero."""
    ctx = Context()
    from xdsl.dialects import arith

    ctx.load_dialect(arith.Arith)

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    # Create constants: 0 and 5
    zero_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)
    builder.insert(zero_const)
    builder.insert(five_const)

    # Create an AddiOp: %result = arith.addi %zero, %five : i32
    # Adding zero should fold to just the other operand
    addi_op = AddiOp(zero_const.result, five_const.result)
    builder.insert(addi_op)

    # Try to fold the operation
    result = builder.try_fold(ctx, addi_op)

    # Should successfully fold
    assert result is not None
    values, new_ops = result
    assert len(values) == 1
    assert len(new_ops) == 1
    assert isinstance(new_op := new_ops[0], ConstantOp)
    assert isinstance(new_op.value, IntegerAttr)
    assert new_op.value.value.data == 6


def test_try_fold_unfoldable_operation():
    """Test that try_fold returns None for operations that don't support folding."""
    ctx = Context()

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    # Create a TestOp which doesn't implement folding
    const_op = ConstantOp.from_int_and_width(5, i32)
    builder.insert(const_op)
    test_op = TestOp((const_op.result,), result_types=(i32,))
    builder.insert(test_op)

    # Try to fold the TestOp
    result = builder.try_fold(ctx, test_op)

    # Should return None since TestOp doesn't implement folding
    assert result is None
