import pytest

from xdsl.builder import Builder
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IntAttr, i32, i64
from xdsl.dialects.scf import If
from xdsl.ir import Block, BlockArgument, Operation, Region
from xdsl.rewriter import InsertPoint


def test_insertion_point_constructors():
    target = Block(
        [
            (op1 := Constant.from_int_and_width(0, 1)),
            (op2 := Constant.from_int_and_width(1, 1)),
        ]
    )

    assert InsertPoint.at_start(target) == InsertPoint(target, op1)
    assert Builder.at_start(target).insertion_point == InsertPoint(target, op1)

    assert InsertPoint.at_end(target) == InsertPoint(target, None)
    assert Builder.at_end(target).insertion_point == InsertPoint(target, None)

    assert InsertPoint.before(op1) == InsertPoint(target, op1)
    assert Builder.before(op1).insertion_point == InsertPoint(target, op1)

    assert InsertPoint.after(op1) == InsertPoint(target, op2)
    assert Builder.after(op1).insertion_point == InsertPoint(target, op2)

    assert InsertPoint.before(op2) == InsertPoint(target, op2)
    assert Builder.before(op2).insertion_point == InsertPoint(target, op2)

    assert InsertPoint.after(op2) == InsertPoint(target, None)
    assert Builder.after(op2).insertion_point == InsertPoint(target, None)


def test_builder():
    target = Block(
        [
            Constant.from_int_and_width(0, 1),
            Constant.from_int_and_width(1, 1),
        ]
    )

    block = Block()
    b = Builder.at_end(block)

    x = Constant.from_int_and_width(0, 1)
    y = Constant.from_int_and_width(1, 1)

    b.insert(x)
    b.insert(y)

    assert target.is_structurally_equivalent(block)


def test_builder_insertion_point():
    target = Block(
        [
            Constant.from_int_and_width(1, 8),
            Constant.from_int_and_width(2, 8),
            Constant.from_int_and_width(3, 8),
        ]
    )

    block = Block()
    b = Builder.at_end(block)

    x = Constant.from_int_and_width(1, 8)
    y = Constant.from_int_and_width(2, 8)
    z = Constant.from_int_and_width(3, 8)

    b.insert(x)
    b.insert(z)

    b.insertion_point = InsertPoint.before(z)

    b.insert(y)

    assert target.is_structurally_equivalent(block)


def test_builder_create_block():
    block1 = Block()
    block2 = Block()
    target = Region([block1, block2])
    builder = Builder.at_start(block1)

    new_block1 = builder.create_block_at_start(target, (i32,))
    assert len(new_block1.args) == 1
    assert new_block1.args[0].type == i32
    assert len(target.blocks) == 3
    assert target.blocks[0] == new_block1
    assert builder.insertion_point == InsertPoint.at_start(new_block1)

    new_block2 = builder.create_block_at_end(target, (i64,))
    assert len(new_block2.args) == 1
    assert new_block2.args[0].type == i64
    assert len(target.blocks) == 4
    assert target.blocks[3] == new_block2
    assert builder.insertion_point == InsertPoint.at_start(new_block2)

    new_block3 = builder.create_block_before(block2, (i32, i64))
    assert len(new_block3.args) == 2
    assert new_block3.args[0].type == i32
    assert new_block3.args[1].type == i64
    assert len(target.blocks) == 5
    assert target.blocks[2] == new_block3
    assert builder.insertion_point == InsertPoint.at_start(new_block3)

    new_block4 = builder.create_block_after(block2, (i64, i32))
    assert len(new_block4.args) == 2
    assert new_block4.args[0].type == i64
    assert new_block4.args[1].type == i32
    assert len(target.blocks) == 6
    assert target.blocks[4] == new_block4
    assert builder.insertion_point == InsertPoint.at_start(new_block4)


def test_builder_listener_op_insert():
    block = Block()
    b = Builder.at_end(block)

    x = Constant.from_int_and_width(1, 32)
    y = Constant.from_int_and_width(2, 32)
    z = Constant.from_int_and_width(3, 32)

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
    b = Builder.at_start(block)

    created_blocks: list[Block] = []

    def add_block_on_create(b: Block):
        created_blocks.append(b)

    b.block_creation_handler = [add_block_on_create]

    b1 = b.create_block_at_start(region)
    b2 = b.create_block_at_end(region)
    b3 = b.create_block_before(block)
    b4 = b.create_block_after(block)

    assert created_blocks == [b1, b2, b3, b4]


def test_build_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                Constant.from_int_and_width(one, i32),
                Constant.from_int_and_width(two, i32),
            ]
        )
    )

    @Builder.region
    def region(b: Builder):
        x = Constant.from_int_and_width(one, i32)
        y = Constant.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    assert target.is_structurally_equivalent(region)


def test_build_callable_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                Constant.from_int_and_width(one, i32),
                Constant.from_int_and_width(two, i32),
            ],
            arg_types=(i32,),
        )
    )

    @Builder.region([i32])
    def region(b: Builder, args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        x = Constant.from_int_and_width(one, i32)
        y = Constant.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    assert target.is_structurally_equivalent(region)


def test_build_implicit_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                Constant.from_int_and_width(one, i32),
                Constant.from_int_and_width(two, i32),
            ]
        )
    )

    @Builder.implicit_region
    def region():
        Constant.from_int_and_width(one, i32)
        Constant.from_int_and_width(two, i32)

    assert target.is_structurally_equivalent(region)


def test_build_implicit_callable_region():
    one = IntAttr(1)
    two = IntAttr(2)

    target = Region(
        Block(
            [
                Constant.from_int_and_width(one, i32),
                Constant.from_int_and_width(two, i32),
            ],
            arg_types=(i32,),
        )
    )

    @Builder.implicit_region([i32])
    def region(args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        Constant.from_int_and_width(one, i32)
        Constant.from_int_and_width(two, i32)

    assert target.is_structurally_equivalent(region)


def test_build_nested_implicit_region():
    target = Region(
        Block(
            [
                cond := Constant.from_int_and_width(1, 1),
                If(
                    cond,
                    (),
                    Region(
                        Block(
                            [
                                Constant.from_int_and_width(2, i32),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    @Builder.implicit_region
    def region():
        cond = Constant.from_int_and_width(1, 1).result

        @Builder.implicit_region
        def then():
            _y = Constant.from_int_and_width(2, i32)

        If(cond, (), then)

    assert target.is_structurally_equivalent(region)


def test_build_implicit_region_fail():
    with pytest.raises(ValueError) as e:
        one = IntAttr(1)
        two = IntAttr(2)
        three = IntAttr(3)

        @Builder.implicit_region
        def region():
            cond = Constant.from_int_and_width(1, 1).result

            _x = Constant.from_int_and_width(one, i32)

            @Builder.implicit_region
            def then_0():
                _y = Constant.from_int_and_width(two, i32)

                @Builder.region
                def then_1(b: Builder):
                    b.insert(Constant.from_int_and_width(three, i32))

                If(cond, (), then_1)

            If(cond, (), then_0)

        _ = region
    assert e.value.args[0] == (
        "Cannot insert operation explicitly when an implicit" " builder exists."
    )
