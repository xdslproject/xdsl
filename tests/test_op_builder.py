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

    ops = iter(block.ops)

    assert next(ops) is x
    assert next(ops) is y

    assert next(ops, None) is None


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

    ops = iter(region.ops)

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is one

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is two

    assert next(ops, None) is None


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

    ops = iter(region.ops)

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is one

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is two

    args = region.blocks[0].args

    assert len(args) == 1
    assert args[0].typ == i32

    assert next(ops, None) is None


def test_build_implicit_region():
    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.implicit_region
    def region():
        Constant.from_int_and_width(one, i32)
        Constant.from_int_and_width(two, i32)

    assert len(region.blocks) == 1

    ops = iter(region.ops)

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is one

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is two

    assert next(ops, None) is None


def test_build_implicit_callable_region():
    one = IntAttr(1)
    two = IntAttr(2)

    @Builder.implicit_region([i32])
    def region(args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        Constant.from_int_and_width(one, i32)
        Constant.from_int_and_width(two, i32)

    assert len(region.blocks) == 1

    ops = iter(region.ops)

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is one

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is two

    args = region.blocks[0].args

    assert len(args) == 1
    assert args[0].typ == i32

    assert next(ops, None) is None


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

    ops_0 = iter(region_0.ops)

    op = next(ops_0)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is one

    assert len(op.regions) == 1

    assert len(region_0.blocks) == 1

    region_1 = op.regions[0]

    assert next(ops_0, None) is None

    ops_1 = iter(region_1.ops)

    op = next(ops_1)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is two

    assert len(op.regions) == 0

    assert next(ops_1, None) is None


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
                    "Cannot insert operation explicitly when an implicit"
                    " builder exists."
                )

            y.add_region(region_2)

        x.add_region(region_1)

    assert len(region_0.blocks) == 1

    ops = iter(region_0.ops)

    op = next(ops)
    assert isinstance(op, Constant)
    assert isinstance(op.value, IntegerAttr)
    assert op.value.value is one

    assert len(op.regions) == 1

    assert next(ops, None) is None
