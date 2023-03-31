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

    ops = block.ops

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