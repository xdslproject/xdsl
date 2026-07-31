from xdsl.backend.liveness import VerifyLivenessContext
from xdsl.dialects.test import TestAllocatableOp, TestRegisterType
from xdsl.utils.test_value import create_ssa_value


def test_has_register_constraints_update_liveness_without_regions():
    u = TestRegisterType.unallocated()
    operand = create_ssa_value(u)
    op = TestAllocatableOp([operand], [], [u], [])
    ctx = VerifyLivenessContext(set())
    ctx.process_op(op)
    assert operand in ctx.alive
