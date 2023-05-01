from xdsl.utils.test_value import TestSSAValue
from xdsl.dialects import riscv


def test_add_op():
    a1 = TestSSAValue(riscv.RegisterType(riscv.Register()))
    a2 = TestSSAValue(riscv.RegisterType(riscv.Register()))
    add_op = riscv.AddOp(a1, a2)
    a0 = add_op.rd

    assert a1.typ is add_op.rs1.typ
    assert a2.typ is add_op.rs2.typ
    assert isinstance(a0.typ, riscv.RegisterType)
