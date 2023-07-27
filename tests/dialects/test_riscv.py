import pytest

from xdsl.builder import Builder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32
from xdsl.utils.exceptions import VerifyException
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
    assert a0.type.data == "a0"
    assert a1.type.data == "a1"
    assert a2.type.data == "a2"


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


def test_label_op_with_region():
    @Builder.implicit_region
    def label_region():
        a1_reg = TestSSAValue(riscv.Registers.A1)
        a2_reg = TestSSAValue(riscv.Registers.A2)
        riscv.AddOp(a1_reg, a2_reg, rd=riscv.Registers.A0)

    label_str = "mylabel"
    label_op = riscv.LabelOp(f"{label_str}", region=label_region)

    assert label_op.label.data == label_str

    code = riscv.riscv_code(ModuleOp([label_op]))
    assert code == f"{label_str}:\n    add a0, a1, a2\n"


def test_return_op():
    return_op = riscv.EbreakOp(comment="my comment")

    assert return_op.comment is not None

    assert return_op.comment.data == "my comment"

    code = riscv.riscv_code(ModuleOp([return_op]))
    assert code == "    ebreak                                       # my comment\n"


def test_immediate_i_inst():
    # I-Type - 12-bits immediate
    a1 = TestSSAValue(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        riscv.AddiOp(a1, 1 << 11, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.AddiOp(a1, -(1 << 11) - 2, rd=riscv.Registers.A0)

    riscv.AddiOp(a1, -(1 << 11), rd=riscv.Registers.A0)

    riscv.AddiOp(a1, (1 << 11) - 1, rd=riscv.Registers.A0)

    """
    Special handling for signed immediates for I- and S-Type instructions
    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#signed-immediates-for-i--and-s-type-instructions
    """

    riscv.AddiOp(a1, 0xFFFFFFFFFFFFF800, rd=riscv.Registers.A0)
    riscv.AddiOp(a1, 0xFFFFFFFFFFFFFFFF, rd=riscv.Registers.A0)
    riscv.AddiOp(a1, 0xFFFFF800, rd=riscv.Registers.A0)
    riscv.AddiOp(a1, 0xFFFFFFFF, rd=riscv.Registers.A0)


def test_immediate_s_inst():
    # S-Type - 12-bits immediate
    a1 = TestSSAValue(riscv.Registers.A1)
    a2 = TestSSAValue(riscv.Registers.A2)

    with pytest.raises(VerifyException):
        riscv.SwOp(a1, a2, 1 << 11)

    with pytest.raises(VerifyException):
        riscv.SwOp(a1, a2, -(1 << 11) - 2)

    riscv.SwOp(a1, a2, -(1 << 11))
    riscv.SwOp(a1, a2, (1 << 11) - 1)

    """
    Special handling for signed immediates for I- and S-Type instructions
    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#signed-immediates-for-i--and-s-type-instructions
    """

    riscv.SwOp(a1, a2, 0xFFFFFFFFFFFFF800)
    riscv.SwOp(a1, a2, 0xFFFFFFFFFFFFFFFF)
    riscv.SwOp(a1, a2, 0xFFFFF800)
    riscv.SwOp(a1, a2, 0xFFFFFFFF)


def test_immediate_u_j_inst():
    # U-Type and J-Type - 20-bits immediate
    with pytest.raises(VerifyException):
        riscv.LuiOp(1 << 20)

    with pytest.raises(VerifyException):
        riscv.LuiOp(-(1 << 20) - 2)

    riscv.LuiOp((1 << 20) - 1)


def test_immediate_jalr_inst():
    # Jalr - 12-bits immediate
    a1 = TestSSAValue(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        riscv.JalrOp(a1, 1 << 12, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.JalrOp(a1, -(1 << 12) - 2, rd=riscv.Registers.A0)

    riscv.JalrOp(a1, (1 << 11) - 1, rd=riscv.Registers.A0)


def test_immediate_pseudo_inst():
    # Pseudo-Instruction with custom handling
    with pytest.raises(VerifyException):
        riscv.LiOp(-(1 << 31) - 1, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.LiOp(1 << 32, rd=riscv.Registers.A0)

    riscv.LiOp((1 << 31) - 1, rd=riscv.Registers.A0)


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
        riscv.FAddSOp(a1, a2).verify()

    f1 = TestSSAValue(riscv.Registers.FT0)
    f2 = TestSSAValue(riscv.Registers.FT1)
    riscv.FAddSOp(f1, f2).verify()
