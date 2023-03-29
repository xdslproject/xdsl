import pytest

from xdsl.builder import Builder
from xdsl.ir import BlockArgument, Block
from xdsl.dialects.builtin import IntAttr, i32, IntegerAttr
from xdsl.dialects.arith import Constant


def test_builder():
    block = Block()
    b = Builder(block)

    x = Constant.from_int_and_width(1, 1)
    y = Constant.from_int_and_width(2, 1)

    b.insert(x)
    b.insert(y)

    ops = list(block.iter_ops())

    assert len(ops) == 2
    assert ops[0] is x
    assert ops[1] is y


def test_build_region():

    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.region
    def region(b: Builder):
        x = Constant.from_int_and_width(one, i32)
        y = Constant.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    assert len(region.blocks) == 1

    ops = region.ops

    assert len(ops) == 2

    assert isinstance(ops[0], Constant)
    assert isinstance(ops[0].value, IntegerAttr)
    assert ops[0].value.value is one

    assert isinstance(ops[1], Constant)
    assert isinstance(ops[1].value, IntegerAttr)
    assert ops[1].value.value is two


def test_build_callable_region():

    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.region([i32])
    def region(b: Builder, args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        x = Constant.from_int_and_width(one, i32)
        y = Constant.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    assert len(region.blocks) == 1

    ops = region.ops

    assert len(ops) == 2

    assert isinstance(ops[0], Constant)
    assert isinstance(ops[0].value, IntegerAttr)
    assert ops[0].value.value is one

    assert isinstance(ops[1], Constant)
    assert isinstance(ops[1].value, IntegerAttr)
    assert ops[1].value.value is two

    args = region.blocks[0].args

    assert len(args) == 1
    assert args[0].typ == i32


def test_build_implicit_region():

    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.implicit_region
    def region():
        Constant.from_int_and_width(one, i32)
        Constant.from_int_and_width(two, i32)

    assert len(region.blocks) == 1

    ops = region.ops

    assert len(ops) == 2

    assert isinstance(ops[0], Constant)
    assert isinstance(ops[0].value, IntegerAttr)
    assert ops[0].value.value is one

    assert isinstance(ops[1], Constant)
    assert isinstance(ops[1].value, IntegerAttr)
    assert ops[1].value.value is two


def test_build_implicit_callable_region():

    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.implicit_region([i32])
    def region(args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        Constant.from_int_and_width(one, i32)
        Constant.from_int_and_width(two, i32)

    assert len(region.blocks) == 1

    ops = region.ops

    assert len(ops) == 2

    assert isinstance(ops[0], Constant)
    assert isinstance(ops[0].value, IntegerAttr)
    assert ops[0].value.value is one

    assert isinstance(ops[1], Constant)
    assert isinstance(ops[1].value, IntegerAttr)
    assert ops[1].value.value is two

    args = region.blocks[0].args

    assert len(args) == 1
    assert args[0].typ == i32


def test_build_nested_implicit_region():

    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.implicit_region
    def region_0():
        x = Constant.from_int_and_width(one, i32)

        @Builder.implicit_region
        def region_1():
            _y = Constant.from_int_and_width(two, i32)

        x.add_region(region_1)

    assert len(region_0.blocks) == 1

    ops_0 = region_0.ops

    assert len(ops_0) == 1

    assert isinstance(ops_0[0], Constant)
    assert isinstance(ops_0[0].value, IntegerAttr)
    assert ops_0[0].value.value is one

    assert len(ops_0[0].regions) == 1

    assert len(region_0.blocks) == 1

    region_1 = ops_0[0].regions[0]

    ops_1 = region_1.ops

    assert len(ops_1) == 1

    assert isinstance(ops_1[0], Constant)
    assert isinstance(ops_1[0].value, IntegerAttr)
    assert ops_1[0].value.value is two

    assert len(ops_1[0].regions) == 0


def test_build_implicit_region_fail():

    one = IntAttr(1)
    two = IntAttr(2)
    three = IntAttr(3)

    @Builder.implicit_region
    def region_0():
        x = Constant.from_int_and_width(one, i32)

        @Builder.implicit_region
        def region_1():
            y = Constant.from_int_and_width(two, i32)

            @Builder.region
            def region_2(b: Builder):
                with pytest.raises(ValueError) as e:
                    b.insert(Constant.from_int_and_width(three, i32))

                assert e.value.args[0] == (
                    'Cannot insert operation explicitly when an implicit'
                    ' builder exists.')

            y.add_region(region_2)

        x.add_region(region_1)

    assert len(region_0.blocks) == 1

    ops = region_0.ops

    assert len(ops) == 1

    assert isinstance(ops[0], Constant)
    assert isinstance(ops[0].value, IntegerAttr)
    assert ops[0].value.value is one

    assert len(ops[0].regions) == 1
