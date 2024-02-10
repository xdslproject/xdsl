from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    ModuleOp,
)
from xdsl.interactive.get_rewrites_and_passes import (
    ALL_PATTERNS,
    IndexedIndividualRewrite,
    IndividualRewrite,
    get_all_possible_rewrites,
)
from xdsl.ir import Block, Region
from xdsl.transforms import individual_rewrite


def test_get_all_possible_rewrite():
    # build module
    index = IndexType()
    module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(module.body):
        function = func.FuncOp("hello", ((index,), (index,)))
        with ImplicitBuilder(function.body) as (n,):
            two = arith.Constant(IntegerAttr(0, index)).result
            res = arith.Addi(two, n)
            func.Return(res)

    expected_res = (
        (
            IndexedIndividualRewrite(
                3, IndividualRewrite(operation="arith.addi", pattern="AddImmediateZero")
            )
        ),
    )

    res = get_all_possible_rewrites(
        ALL_PATTERNS, module, individual_rewrite.REWRITE_BY_NAMES
    )
    assert res == expected_res


def test_empty_get_all_possible_rewrite():
    # build module
    index = IndexType()
    module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(module.body):
        function = func.FuncOp("hello", ((index,), (index,)))
        with ImplicitBuilder(function.body) as (n,):
            two = arith.Constant(IntegerAttr(2, index)).result
            three = arith.Constant(IntegerAttr(2, index)).result
            res_1 = arith.Muli(n, two)
            res_2 = arith.Muli(n, three)
            res = arith.Muli(res_1, res_2)
            func.Return(res)

    expected_res = ()

    res = get_all_possible_rewrites(
        ALL_PATTERNS, module, individual_rewrite.REWRITE_BY_NAMES
    )
    assert res == expected_res
