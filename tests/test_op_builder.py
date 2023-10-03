import pytest

from xdsl.builder import Builder
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IntAttr, i32
from xdsl.dialects.scf import If
from xdsl.ir import Block, BlockArgument, Region


def test_builder():
    target = Block(
        [
            Constant.from_int_and_width(1, 1),
            Constant.from_int_and_width(2, 1),
        ]
    )

    block = Block()
    b = Builder(block)

    x = Constant.from_int_and_width(1, 1)
    y = Constant.from_int_and_width(2, 1)

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
    b = Builder(block)

    x = Constant.from_int_and_width(1, 8)
    y = Constant.from_int_and_width(2, 8)
    z = Constant.from_int_and_width(3, 8)

    b.insert(x)
    b.insert(z)

    b.insertion_point = z

    b.insert(y)

    assert target.is_structurally_equivalent(block)

    with pytest.raises(ValueError) as e:
        b.insertion_point = Constant.from_int_and_width(4, 8)

    assert e.value.args[0] == "Insertion point must be in the builder's `block`"

    with pytest.raises(ValueError) as e:
        Builder(block, Constant.from_int_and_width(4, 8))

    assert e.value.args[0] == "Insertion point must be in the builder's `block`"


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
