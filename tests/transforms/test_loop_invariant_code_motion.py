from xdsl.builder import ImplicitBuilder
from xdsl.dialects import test
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block, Region
from xdsl.transforms.loop_invariant_code_motion import can_be_hoisted

index = IndexType()


def test_can_be_hoisted():
    """Test for with loop-variant variables"""

    outer_region = Region(Block(arg_types=(index,)))
    mid_region = Region(Block(arg_types=(index,)))
    inner_region = Region(Block(arg_types=(index,)))

    with ImplicitBuilder(outer_region) as (outer_arg,):
        outer_op_no_args = test.TestOp()
        outer_op_outer_arg = test.TestOp(operands=(outer_arg,))
        with ImplicitBuilder(mid_region) as (mid_arg,):
            mid_op_no_args = test.TestOp()
            mid_op_outer_arg = test.TestOp(operands=(outer_arg,))
            mid_op_mid_arg = test.TestOp(operands=(mid_arg,))
            with ImplicitBuilder(inner_region) as (inner_arg,):
                inner_op_no_args = test.TestOp()
                inner_op_outer_arg = test.TestOp(operands=(outer_arg,))
                inner_op_inner_arg = test.TestOp(operands=(inner_arg,))
                inner_term_op = test.TestTermOp()
            mid_term_op = test.TestTermOp()
            inner_region_op = test.TestOp(regions=(inner_region,))
        mid_region_op = test.TestOp(regions=(mid_region,))
        out_term_op = test.TestTermOp()

    # outer region
    assert can_be_hoisted(outer_op_no_args, outer_region)
    assert not can_be_hoisted(outer_op_outer_arg, outer_region)
    assert can_be_hoisted(mid_op_no_args, outer_region)
    assert not can_be_hoisted(mid_op_outer_arg, outer_region)
    assert not can_be_hoisted(mid_op_mid_arg, outer_region)
    assert can_be_hoisted(inner_op_no_args, outer_region)
    assert not can_be_hoisted(inner_op_outer_arg, outer_region)
    assert not can_be_hoisted(inner_op_inner_arg, outer_region)
    assert not can_be_hoisted(inner_term_op, outer_region)
    assert not can_be_hoisted(mid_term_op, outer_region)
    assert not can_be_hoisted(inner_region_op, outer_region)
    assert not can_be_hoisted(mid_region_op, outer_region)
    assert not can_be_hoisted(out_term_op, outer_region)

    # mid region
    assert can_be_hoisted(outer_op_no_args, mid_region)
    assert can_be_hoisted(outer_op_outer_arg, mid_region)
    assert can_be_hoisted(mid_op_no_args, mid_region)
    assert can_be_hoisted(mid_op_outer_arg, mid_region)
    assert not can_be_hoisted(mid_op_mid_arg, mid_region)
    assert can_be_hoisted(inner_op_no_args, mid_region)
    assert can_be_hoisted(inner_op_outer_arg, mid_region)
    assert not can_be_hoisted(inner_op_inner_arg, mid_region)
    assert not can_be_hoisted(inner_term_op, mid_region)
    assert not can_be_hoisted(mid_term_op, mid_region)
    assert can_be_hoisted(inner_region_op, mid_region)
    assert can_be_hoisted(mid_region_op, mid_region)
    assert not can_be_hoisted(out_term_op, mid_region)

    # inner region
    assert can_be_hoisted(outer_op_no_args, inner_region)
    assert can_be_hoisted(outer_op_outer_arg, inner_region)
    assert can_be_hoisted(mid_op_no_args, inner_region)
    assert can_be_hoisted(mid_op_outer_arg, inner_region)
    assert can_be_hoisted(mid_op_mid_arg, inner_region)
    assert can_be_hoisted(inner_op_no_args, inner_region)
    assert can_be_hoisted(inner_op_outer_arg, inner_region)
    assert not can_be_hoisted(inner_op_inner_arg, inner_region)
    assert not can_be_hoisted(inner_term_op, inner_region)
    assert not can_be_hoisted(mid_term_op, inner_region)
    assert can_be_hoisted(inner_region_op, inner_region)
    assert can_be_hoisted(mid_region_op, inner_region)
    assert not can_be_hoisted(out_term_op, inner_region)
