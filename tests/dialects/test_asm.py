from xdsl.dialects import asm, x86
from xdsl.dialects.builtin import i64
from xdsl.utils.test_value import create_ssa_value


def test_from_reg_init():
    reg = create_ssa_value(x86.registers.UNALLOCATED_REG64)
    op = asm.FromRegOp(reg, i64)

    assert op.register == reg
    assert op.value.type == i64


def test_to_reg_init():
    val = create_ssa_value(i64)
    reg_type = x86.registers.UNALLOCATED_REG64
    op = asm.ToRegOp(val, reg_type)

    assert op.value == val
    assert op.register.type == reg_type
