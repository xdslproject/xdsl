from io import StringIO
from xdsl.builder import Builder
from xdsl.utils.test_value import TestSSAValue
from xdsl.dialects import riscv

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32

from xdsl.utils.exceptions import VerifyException

import pytest


def test_add_op():
    a1 = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
    a2 = TestSSAValue(riscv.RegisterType(riscv.Registers.A2))
    add_op = riscv.AddOp(a1, a2, rd=riscv.Registers.A0)
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
    a1 = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
    zero = TestSSAValue(riscv.RegisterType(riscv.Registers.ZERO))
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


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    riscv.print_assembly(module, stream)
    return stream.getvalue()


def test_comment_op():
    comment_op = riscv.CommentOp("my comment")

    assert comment_op.comment.data == "my comment"

    code = riscv_code(ModuleOp([comment_op]))
    assert code == "    # my comment\n"


def test_label_op_without_comment():
    label_str = "mylabel"
    label_op = riscv.LabelOp(label_str)

    assert label_op.label.data == label_str

    code = riscv_code(ModuleOp([label_op]))
    assert code == f"{label_str}:\n"


def test_label_op_with_comment():
    label_str = "mylabel"
    label_op = riscv.LabelOp(f"{label_str}", comment="my label")

    assert label_op.label.data == label_str

    code = riscv_code(ModuleOp([label_op]))
    assert code == f"{label_str}:                                         # my label\n"


def test_label_op_with_region():
    @Builder.implicit_region
    def label_region():
        a1_reg = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
        a2_reg = TestSSAValue(riscv.RegisterType(riscv.Registers.A2))
        riscv.AddOp(a1_reg, a2_reg, rd=riscv.Registers.A0)

    label_str = "mylabel"
    label_op = riscv.LabelOp(f"{label_str}", region=label_region)

    assert label_op.label.data == label_str

    code = riscv_code(ModuleOp([label_op]))
    assert code == f"{label_str}:\n    add a0, a1, a2\n"


def test_return_op():
    return_op = riscv.EbreakOp(comment="my comment")

    assert return_op.comment is not None

    assert return_op.comment.data == "my comment"

    code = riscv_code(ModuleOp([return_op]))
    assert code == "    ebreak                                       # my comment\n"


def test_immediate_i_inst():
    # I-Type - 12-bits immediate
    a1 = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))

    with pytest.raises(VerifyException):
        pos_invalid_addi_op = riscv.AddiOp(a1, 1 << 11, rd=riscv.Registers.A0)
        pos_invalid_addi_op.verify()

    with pytest.raises(VerifyException):
        neg_invalid_addi_op = riscv.AddiOp(a1, -(1 << 11) - 2, rd=riscv.Registers.A0)
        neg_invalid_addi_op.verify()

    neg_valid_addi_op = riscv.AddiOp(a1, -(1 << 11), rd=riscv.Registers.A0)
    neg_valid_addi_op.verify()

    pos_valid_addi_op = riscv.AddiOp(a1, (1 << 11) - 1, rd=riscv.Registers.A0)
    pos_valid_addi_op.verify()

    """
    Special handling for signed immediates for I- and S-Type instructions
    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#signed-immediates-for-i--and-s-type-instructions
    """

    sext_0_valid_addi_op = riscv.AddiOp(a1, 0xFFFFFFFFFFFFF800, rd=riscv.Registers.A0)
    sext_0_valid_addi_op.verify()

    sext_1_valid_addi_op = riscv.AddiOp(a1, 0xFFFFFFFFFFFFFFFF, rd=riscv.Registers.A0)
    sext_1_valid_addi_op.verify()

    sext_2_valid_addi_op = riscv.AddiOp(a1, 0xFFFFF800, rd=riscv.Registers.A0)
    sext_2_valid_addi_op.verify()

    sext_3_valid_addi_op = riscv.AddiOp(a1, 0xFFFFFFFF, rd=riscv.Registers.A0)
    sext_3_valid_addi_op.verify()


def test_immediate_s_inst():
    # S-Type - 12-bits immediate
    a1 = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
    a2 = TestSSAValue(riscv.RegisterType(riscv.Registers.A2))

    with pytest.raises(VerifyException):
        pos_invalid_sw_op = riscv.SwOp(a1, a2, 1 << 11)
        pos_invalid_sw_op.verify()

    with pytest.raises(VerifyException):
        neg_invalid_sw_op = riscv.SwOp(a1, a2, -(1 << 11) - 2)
        neg_invalid_sw_op.verify()

    neg_valid_sw_op = riscv.SwOp(a1, a2, -(1 << 11))
    neg_valid_sw_op.verify()

    pos_valid_sw_op = riscv.SwOp(a1, a2, (1 << 11) - 1)
    pos_valid_sw_op.verify()

    """
    Special handling for signed immediates for I- and S-Type instructions
    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#signed-immediates-for-i--and-s-type-instructions
    """

    sext_0_valid_sw_op = riscv.SwOp(a1, a2, 0xFFFFFFFFFFFFF800)
    sext_0_valid_sw_op.verify()

    sext_1_valid_sw_op = riscv.SwOp(a1, a2, 0xFFFFFFFFFFFFFFFF)
    sext_1_valid_sw_op.verify()

    sext_2_valid_sw_op = riscv.SwOp(a1, a2, 0xFFFFF800)
    sext_2_valid_sw_op.verify()

    sext_3_valid_sw_op = riscv.SwOp(a1, a2, 0xFFFFFFFF)
    sext_3_valid_sw_op.verify()


def test_immediate_u_j_inst():
    # U-Type and J-Type - 20-bits immediate
    with pytest.raises(VerifyException):
        pos_invalid_j_op = riscv.LuiOp(1 << 20)
        pos_invalid_j_op.verify()

    with pytest.raises(VerifyException):
        neg_invalid_j_op = riscv.LuiOp(-(1 << 20) - 2)
        neg_invalid_j_op.verify()

    valid_j_op = riscv.LuiOp((1 << 20) - 1)
    valid_j_op.verify()


def test_immediate_jalr_inst():
    # Jalr - 12-bits immediate
    a1 = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))

    with pytest.raises(VerifyException):
        pos_invalid_jalr_op = riscv.JalrOp(a1, 1 << 12, rd=riscv.Registers.A0)
        pos_invalid_jalr_op.verify()

    with pytest.raises(VerifyException):
        neg_invalid_jalr_op = riscv.JalrOp(a1, -(1 << 12) - 2, rd=riscv.Registers.A0)
        neg_invalid_jalr_op.verify()

    jalr_op = riscv.JalrOp(a1, (1 << 11) - 1, rd=riscv.Registers.A0)
    jalr_op.verify()


def test_immediate_pseudo_inst():
    # Pseudo-Instruction with custom handling
    with pytest.raises(VerifyException):
        neg_invalid_li_op = riscv.LiOp(-(1 << 31) - 1, rd=riscv.Registers.A0)
        neg_invalid_li_op.verify()

    with pytest.raises(VerifyException):
        pos_invalid_li_op = riscv.LiOp(1 << 32, rd=riscv.Registers.A0)
        pos_invalid_li_op.verify()

    valid_li_op = riscv.LiOp((1 << 31) - 1, rd=riscv.Registers.A0)
    valid_li_op.verify()


def test_immediate_shift_inst():
    # Shift instructions (SLLI, SRLI, SRAI) - 5-bits immediate
    a1 = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))

    with pytest.raises(VerifyException):
        pos_invalid_slli_op = riscv.SlliOp(a1, 1 << 5, rd=riscv.Registers.A0)
        pos_invalid_slli_op.verify()

    with pytest.raises(VerifyException):
        neg_invalid_slli_op = riscv.SlliOp(a1, -1, rd=riscv.Registers.A0)
        neg_invalid_slli_op.verify()

    valid_slli_op = riscv.SlliOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)
    valid_slli_op.verify()
