import pytest

from xdsl.context import MLContext
from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    IntAttr,
    IntegerAttr,
    ModuleOp,
    NoneAttr,
    Signedness,
    i32,
)
from xdsl.parser import Parser
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_add_op():
    a1 = TestSSAValue(riscv.Registers.A1)
    a2 = TestSSAValue(riscv.Registers.A2)
    add_op = riscv.AddOp(a1, a2, rd=riscv.Registers.A0)
    a0 = add_op.rd

    assert a1.type is add_op.rs1.type
    assert a2.type is add_op.rs2.type
    assert isinstance(a0.type, riscv.IntRegisterType)
    assert isinstance(a1.type, riscv.IntRegisterType)
    assert isinstance(a2.type, riscv.IntRegisterType)
    assert a0.type.spelling.data == "a0"
    assert a0.type.index == IntAttr(10)
    assert a1.type.spelling.data == "a1"
    assert a1.type.index == IntAttr(11)
    assert a2.type.spelling.data == "a2"
    assert a2.type.index == IntAttr(12)

    # Registers that aren't predefined should not have an index.
    assert isinstance(riscv.IntRegisterType("j1").index, NoneAttr)


def test_csr_op():
    a1 = TestSSAValue(riscv.Registers.A1)
    zero = TestSSAValue(riscv.Registers.ZERO)
    csr = IntegerAttr(16, i32)
    # CsrrwOp
    riscv.CsrrwOp(rs1=a1, csr=csr, rd=riscv.Registers.A2).verify()
    riscv.CsrrwOp(rs1=a1, csr=csr, rd=riscv.Registers.ZERO).verify()
    riscv.CsrrwOp(rs1=a1, csr=csr, writeonly=True, rd=riscv.Registers.ZERO).verify()
    with pytest.raises(VerifyException):
        riscv.CsrrwOp(rs1=a1, csr=csr, writeonly=True, rd=riscv.Registers.A2).verify()
    # CsrrsOp
    riscv.CsrrsOp(rs1=a1, csr=csr, rd=riscv.Registers.A2).verify()
    riscv.CsrrsOp(rs1=zero, csr=csr, rd=riscv.Registers.A2).verify()
    riscv.CsrrsOp(rs1=zero, csr=csr, readonly=True, rd=riscv.Registers.A2).verify()
    with pytest.raises(VerifyException):
        riscv.CsrrsOp(rs1=a1, csr=csr, readonly=True, rd=riscv.Registers.A2).verify()
    # CsrrcOp
    riscv.CsrrcOp(rs1=a1, csr=csr, rd=riscv.Registers.A2).verify()
    riscv.CsrrcOp(rs1=zero, csr=csr, rd=riscv.Registers.A2).verify()
    riscv.CsrrcOp(rs1=zero, csr=csr, readonly=True, rd=riscv.Registers.A2).verify()
    with pytest.raises(VerifyException):
        riscv.CsrrcOp(rs1=a1, csr=csr, readonly=True, rd=riscv.Registers.A2).verify()
    # CsrrwiOp
    riscv.CsrrwiOp(
        csr=csr, immediate=IntegerAttr(0, i32), rd=riscv.Registers.A2
    ).verify()
    riscv.CsrrwiOp(
        csr=csr, immediate=IntegerAttr(0, i32), rd=riscv.Registers.ZERO
    ).verify()
    riscv.CsrrwiOp(
        csr=csr, immediate=IntegerAttr(0, i32), writeonly=True, rd=riscv.Registers.ZERO
    ).verify()
    with pytest.raises(VerifyException):
        riscv.CsrrwiOp(
            csr=csr,
            immediate=IntegerAttr(0, i32),
            writeonly=True,
            rd=riscv.Registers.A2,
        ).verify()
    # CsrrsiOp
    riscv.CsrrsiOp(
        csr=csr, immediate=IntegerAttr(0, i32), rd=riscv.Registers.A2
    ).verify()
    riscv.CsrrsiOp(
        csr=csr, immediate=IntegerAttr(1, i32), rd=riscv.Registers.A2
    ).verify()
    # CsrrciOp
    riscv.CsrrciOp(
        csr=csr, immediate=IntegerAttr(0, i32), rd=riscv.Registers.A2
    ).verify()
    riscv.CsrrsiOp(
        csr=csr, immediate=IntegerAttr(1, i32), rd=riscv.Registers.A2
    ).verify()


def test_comment_op():
    comment_op = riscv.CommentOp("my comment")

    assert comment_op.comment.data == "my comment"

    code = riscv.riscv_code(ModuleOp([comment_op]))
    assert code == "    # my comment\n"


def test_label_op_without_comment():
    label_str = "mylabel"
    label_op = riscv.LabelOp(label_str)

    assert label_op.label.data == label_str

    code = riscv.riscv_code(ModuleOp([label_op]))
    assert code == f"{label_str}:\n"


def test_label_op_with_comment():
    label_str = "mylabel"
    label_op = riscv.LabelOp(f"{label_str}", comment="my label")

    assert label_op.label.data == label_str

    code = riscv.riscv_code(ModuleOp([label_op]))
    assert code == f"{label_str}:                                         # my label\n"


def test_return_op():
    return_op = riscv.EbreakOp(comment="my comment")

    assert return_op.comment is not None

    assert return_op.comment.data == "my comment"

    code = riscv.riscv_code(ModuleOp([return_op]))
    assert code == "    ebreak                                       # my comment\n"


def test_immediate_i_inst():
    # I-Type - 12-bits signed immediate
    lb, ub = Signedness.SIGNED.value_range(12)
    a1 = TestSSAValue(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        riscv.AddiOp(a1, ub, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.AddiOp(a1, lb - 1, rd=riscv.Registers.A0)

    riscv.AddiOp(a1, ub - 1, rd=riscv.Registers.A0)
    riscv.AddiOp(a1, lb, rd=riscv.Registers.A0)


def test_immediate_s_inst():
    # S-Type - 12-bits signed immediate
    lb, ub = Signedness.SIGNED.value_range(12)
    a1 = TestSSAValue(riscv.Registers.A1)
    a2 = TestSSAValue(riscv.Registers.A2)

    with pytest.raises(VerifyException):
        riscv.SwOp(a1, a2, ub)

    with pytest.raises(VerifyException):
        riscv.SwOp(a1, a2, lb - 1)

    riscv.SwOp(a1, a2, ub - 1)
    riscv.SwOp(a1, a2, lb)


def test_immediate_u_j_inst():
    # U-Type and J-Type - 20-bits immediate
    lb, ub = Signedness.SIGNLESS.value_range(20)
    assert ub == 1048576
    assert lb == -524288

    with pytest.raises(VerifyException):
        riscv.LuiOp(ub)

    with pytest.raises(VerifyException):
        riscv.LuiOp(lb - 1)

    riscv.LuiOp(ub - 1)
    riscv.LuiOp(lb)


def test_immediate_jalr_inst():
    # Jalr - 12-bits immediate
    a1 = TestSSAValue(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        riscv.JalrOp(a1, 1 << 12, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.JalrOp(a1, -(1 << 12) - 2, rd=riscv.Registers.A0)

    riscv.JalrOp(a1, (1 << 11) - 1, rd=riscv.Registers.A0)


def test_immediate_pseudo_inst():
    lb, ub = Signedness.SIGNLESS.value_range(32)
    assert ub == 4294967296
    assert lb == -2147483648

    # Pseudo-Instruction with custom handling
    with pytest.raises(VerifyException):
        riscv.LiOp(ub, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.LiOp(lb - 1, rd=riscv.Registers.A0)

    riscv.LiOp(ub - 1, rd=riscv.Registers.A0)
    riscv.LiOp(lb, rd=riscv.Registers.A0)


def test_immediate_shift_inst():
    # Shift instructions (SLLI, SRLI, SRAI) - 5-bits immediate
    a1 = TestSSAValue(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        riscv.SlliOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.SlliOp(a1, -1, rd=riscv.Registers.A0)

    riscv.SlliOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_float_register():
    with pytest.raises(VerifyException, match="not in"):
        riscv.IntRegisterType("ft9")
    with pytest.raises(VerifyException, match="not in"):
        riscv.FloatRegisterType("a0")

    a1 = TestSSAValue(riscv.Registers.A1)
    a2 = TestSSAValue(riscv.Registers.A2)
    with pytest.raises(VerifyException, match="Operation does not verify"):
        riscv.FAddSOp(a1, a2, rd=riscv.FloatRegisterType.unallocated()).verify()

    f1 = TestSSAValue(riscv.Registers.FT0)
    f2 = TestSSAValue(riscv.Registers.FT1)
    riscv.FAddSOp(f1, f2, rd=riscv.FloatRegisterType.unallocated()).verify()


def test_riscv_parse_immediate_value():
    ctx = MLContext()
    ctx.load_dialect(riscv.RISCV)

    prog = """riscv.jalr %0, 1.1, !riscv.reg : (!riscv.reg) -> ()"""
    parser = Parser(ctx, prog)
    with pytest.raises(ParseError, match="Expected immediate"):
        parser.parse_operation()


def test_asm_section():
    section = riscv.AssemblySectionOp("section")
    section.verify()


def test_get_constant_value():
    li_op = riscv.LiOp(1)
    li_val = get_constant_value(li_op.rd)
    assert li_val == IntegerAttr.from_int_and_width(1, 32)
    zero_op = riscv.GetRegisterOp(riscv.Registers.ZERO)
    zero_val = get_constant_value(zero_op.res)
    assert zero_val == IntegerAttr.from_int_and_width(0, 32)
