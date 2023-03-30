from xdsl.builder import OpBuilder
from xdsl.ir import BlockArgument, Block
from xdsl.dialects.builtin import IntAttr, i32, IntegerAttr
from xdsl.dialects.arith import Constant


def test_builder():
    block = Block()
    b = OpBuilder(block)

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

    @OpBuilder.region
    def region(b: OpBuilder):
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

    @OpBuilder.callable_region(([i32], [i32]))
    def callable_region(b: OpBuilder, args: tuple[BlockArgument, ...]):
        assert len(args) == 1

        x = Constant.from_int_and_width(one, i32)
        y = Constant.from_int_and_width(two, i32)

        b.insert(x)
        b.insert(y)

    region, ftype = callable_region

    assert ftype.inputs.data == (i32, )
    assert ftype.outputs.data == (i32, )

    assert len(region.blocks) == 1

    ops = region.ops

    assert len(ops) == 2

    assert isinstance(ops[0], Constant)
    assert isinstance(ops[0].value, IntegerAttr)
    assert ops[0].value.value is one

    assert isinstance(ops[1], Constant)
    assert isinstance(ops[1].value, IntegerAttr)
    assert ops[1].value.value is two
