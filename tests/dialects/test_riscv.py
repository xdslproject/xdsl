from xdsl.utils.test_value import TestSSAValue
from xdsl.dialects import riscv

from xdsl.dialects.builtin import IntegerAttr, i32

from xdsl.utils.exceptions import VerifyException

import pytest


def test_add_op():
    a1 = TestSSAValue(riscv.RegisterType(riscv.Register("a1")))
    a2 = TestSSAValue(riscv.RegisterType(riscv.Register("a2")))
    add_op = riscv.AddOp(a1, a2, rd="a0")
    a0 = add_op.rd

    assert a1.typ is add_op.rs1.typ
    assert a2.typ is add_op.rs2.typ
    assert isinstance(a0.typ, riscv.RegisterType)
    assert isinstance(a1.typ, riscv.RegisterType)
    assert isinstance(a2.typ, riscv.RegisterType)
    assert a0.typ.data.name == "a0"
    assert a1.typ.data.name == "a1"
    assert a2.typ.data.name == "a2"


def test_csr_op():
    a1 = TestSSAValue(riscv.RegisterType(riscv.Register("a1")))
    zero = TestSSAValue(riscv.RegisterType(riscv.Register("zero")))
    csr = IntegerAttr(16, i32)
    # CsrrwOp
    riscv.CsrrwOp(rs1=a1, csr=csr, writeonly=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrwOp(rs1=a1, csr=csr, writeonly=IntegerAttr(0, i32), rd="zero").verify()
    riscv.CsrrwOp(rs1=a1, csr=csr, writeonly=IntegerAttr(1, i32), rd="zero").verify()
    with pytest.raises(VerifyException):
        riscv.CsrrwOp(rs1=a1, csr=csr, writeonly=IntegerAttr(1, i32), rd="a2").verify()
    # CsrrsOp
    riscv.CsrrsOp(rs1=a1, csr=csr, readonly=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrsOp(rs1=zero, csr=csr, readonly=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrsOp(rs1=zero, csr=csr, readonly=IntegerAttr(1, i32), rd="a2").verify()
    with pytest.raises(VerifyException):
        riscv.CsrrsOp(rs1=a1, csr=csr, readonly=IntegerAttr(1, i32), rd="a2").verify()
    # CsrrcOp
    riscv.CsrrcOp(rs1=a1, csr=csr, readonly=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrcOp(rs1=zero, csr=csr, readonly=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrcOp(rs1=zero, csr=csr, readonly=IntegerAttr(1, i32), rd="a2").verify()
    with pytest.raises(VerifyException):
        riscv.CsrrcOp(rs1=a1, csr=csr, readonly=IntegerAttr(1, i32), rd="a2").verify()
    # CsrrwiOp
    riscv.CsrrwiOp(csr=csr, writeonly=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrwiOp(csr=csr, writeonly=IntegerAttr(0, i32), rd="zero").verify()
    riscv.CsrrwiOp(csr=csr, writeonly=IntegerAttr(1, i32), rd="zero").verify()
    with pytest.raises(VerifyException):
        riscv.CsrrwiOp(csr=csr, writeonly=IntegerAttr(1, i32), rd="a2").verify()
    # CsrrsiOp
    riscv.CsrrsiOp(csr=csr, immediate=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrsiOp(csr=csr, immediate=IntegerAttr(1, i32), rd="a2").verify()
    # CsrrciOp
    riscv.CsrrciOp(csr=csr, immediate=IntegerAttr(0, i32), rd="a2").verify()
    riscv.CsrrsiOp(csr=csr, immediate=IntegerAttr(1, i32), rd="a2").verify()
