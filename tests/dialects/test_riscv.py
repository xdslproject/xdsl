from xdsl.utils.test_value import TestSSAValue
from xdsl.dialects import riscv


def test_add_op():
    a1 = TestSSAValue(riscv.RegisterType(riscv.Register.from_name("a1")))
    a2 = TestSSAValue(riscv.RegisterType(riscv.Register.from_name("a2")))
    add_op = riscv.AddOp(a1, a2, rd="a0")
    a0 = add_op.rd

    assert a1.typ is add_op.rs1.typ
    assert a2.typ is add_op.rs2.typ
    assert isinstance(a0.typ, riscv.RegisterType)
    assert a0.typ.data.abi_name == "a0"

    code = riscv.riscv_code([add_op])
    expected = "    add a0, a1, a2\n"

    assert code == expected
