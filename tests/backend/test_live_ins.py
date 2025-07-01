from ordered_set import OrderedSet

from xdsl.backend.register_allocator import live_ins_per_block
from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import i32
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region, SSAValue


def test_live_ins():
    outer_region = Region(Block(arg_types=(i32, i32)))
    middle_region = Region(Block(arg_types=(i32, i32)))
    inner_region = Region(Block(arg_types=(i32, i32)))
    with ImplicitBuilder(outer_region) as (a, b):
        c, d = TestOp(
            operands=(a, b),
            result_types=(i32, i32),
        ).results
        TestOp(regions=(middle_region,))
        with ImplicitBuilder(middle_region) as (e, f):
            g, h = TestOp(operands=(a, c), result_types=(i32, i32)).results
            TestOp(regions=(inner_region,))
            with ImplicitBuilder(inner_region) as (i, j):
                k, l = TestOp(operands=(b, c, d, g), result_types=(i32, i32)).results
                TestOp(operands=(k,)).results

    name_per_block = {
        outer_region.block: "outer",
        inner_region.block: "inner",
        middle_region.block: "middle",
    }

    name_per_value: dict[SSAValue, str] = {
        a: "a",
        b: "b",
        c: "c",
        d: "d",
        e: "e",
        f: "f",
        g: "g",
        h: "h",
        i: "i",
        j: "j",
        k: "k",
        l: "l",
    }

    # To make test failures easier to read
    def rename(
        mapping: dict[Block, OrderedSet[SSAValue]],
    ) -> dict[str, OrderedSet[str]]:
        return {
            name_per_block[block]: OrderedSet(name_per_value[value] for value in values)
            for block, values in mapping.items()
        }

    reference_live_ins: dict[Block, OrderedSet[SSAValue]] = {
        outer_region.block: OrderedSet([]),
        middle_region.block: OrderedSet([b, c, d, a]),
        inner_region.block: OrderedSet([b, c, d, g]),
    }

    assert rename(live_ins_per_block(inner_region.block)) == rename(
        {inner_region.block: reference_live_ins[inner_region.block]}
    )

    assert rename(live_ins_per_block(middle_region.block)) == rename(
        {
            inner_region.block: reference_live_ins[inner_region.block],
            middle_region.block: reference_live_ins[middle_region.block],
        }
    )

    assert rename(live_ins_per_block(outer_region.block)) == rename(
        {
            inner_region.block: reference_live_ins[inner_region.block],
            middle_region.block: reference_live_ins[middle_region.block],
            outer_region.block: reference_live_ins[outer_region.block],
        }
    )
