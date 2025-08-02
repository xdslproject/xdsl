import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import i1
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.ir.region_utils import used_values_defined_above


def test_used_values_defined_above():
    r = Region(Block())
    r0 = Region(Block())
    r1 = Region(Block())
    r00 = Region(Block())
    r01 = Region(Block())

    with ImplicitBuilder(r.block):
        (v, _) = TestOp(result_types=(i1, i1)).results
        TestOp(regions=(r0, r1))
        with ImplicitBuilder(r0):
            (v0, _) = TestOp(operands=(v,), result_types=(i1, i1)).results
            TestOp(regions=(r00, r01))
            with ImplicitBuilder(r00):
                (v00, _) = TestOp(
                    operands=(
                        v,
                        v0,
                    ),
                    result_types=(i1, i1),
                ).results
                TestOp(operands=(v00,))
            with ImplicitBuilder(r01):
                (v01, _) = TestOp(
                    operands=(
                        v,
                        v0,
                    ),
                    result_types=(i1, i1),
                ).results
                TestOp(operands=(v01,))
        with ImplicitBuilder(r1):
            (v1, _) = TestOp(operands=(v,), result_types=(i1, i1)).results
            TestOp(operands=(v1,))

    assert used_values_defined_above(r) == set()
    assert used_values_defined_above(r0) == {v}
    assert used_values_defined_above(r1) == {v}
    assert used_values_defined_above(r00) == {v, v0}
    assert used_values_defined_above(r01) == {v, v0}

    assert used_values_defined_above(r00, r0) == {v}
    assert used_values_defined_above(r01, r0) == {v}

    with pytest.raises(
        AssertionError,
        match="expected isolation limit to be an ancestor of the given region",
    ):
        used_values_defined_above(r0, r1)
