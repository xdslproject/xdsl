"""
Test HW traits and interfaces.
"""


from xdsl.dialects.hw import (
    InnerSymTarget,
)
from xdsl.dialects.test import TestOp


def test_inner_sym_target():
    invalid_target = InnerSymTarget()
    assert not invalid_target

    operand1 = TestOp()

    target = InnerSymTarget(operand1)
    assert target
    assert target.is_op_only()
    assert not target.is_field()

    sub_target = InnerSymTarget.get_target_for_subfield(target, 1)
    assert isinstance(sub_target, InnerSymTarget)
    assert sub_target
    assert not sub_target.is_op_only()
    assert sub_target.is_field()
