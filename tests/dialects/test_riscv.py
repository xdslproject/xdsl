import pytest

from xdsl.context import Context
from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    IntAttr,
    IntegerAttr,
    ModuleOp,
    Signedness,
    i32,
)
from xdsl.parser import Parser
from xdsl.traits import ConstantLike
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_add_op():
    a1 = create_ssa_value(riscv.Registers.A1)
    a2 = create_ssa_value(riscv.Registers.A2)
    add_op = riscv.AddOp(a1, a2, rd=riscv.Registers.A0)
    a0 = add_op.rd

    assert a1.type is add_op.rs1.type
    assert a2.type is add_op.rs2.type
    assert isinstance(a0.type, riscv.IntRegisterType)
    assert isinstance(a1.type, riscv.IntRegisterType)
    assert isinstance(a2.type, riscv.IntRegisterType)
    assert a0.type.register_name.data == "a0"
    assert a0.type.index == IntAttr(10)
    assert a1.type.register_name.data == "a1"
    assert a1.type.index == IntAttr(11)
    assert a2.type.register_name.data == "a2"
    assert a2.type.index == IntAttr(12)

    # Registers that aren't predefined should not have an index.
    assert riscv.IntRegisterType.infinite_register(1).index == IntAttr(-2)


def test_is_non_zero():
    # Test zero register
    x0_reg = riscv.IntRegisterType.from_name("x0")
    assert not riscv.is_non_zero(riscv.Registers.ZERO)
    assert not riscv.is_non_zero(x0_reg)

    # Test non-zero registers
    a0_reg = riscv.Registers.A0
    t0_reg = riscv.Registers.T0
    assert riscv.is_non_zero(a0_reg)
    assert riscv.is_non_zero(t0_reg)

    # Test unallocated register
    unalloc_reg = riscv.IntRegisterType.unallocated()
    assert not riscv.is_non_zero(unalloc_reg)


def test_csr_op():
    a1 = create_ssa_value(riscv.Registers.A1)
    zero = create_ssa_value(riscv.Registers.ZERO)
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
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        riscv.AddiOp(a1, ub, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        riscv.AddiOp(a1, lb - 1, rd=riscv.Registers.A0)

    riscv.AddiOp(a1, ub - 1, rd=riscv.Registers.A0)
    riscv.AddiOp(a1, lb, rd=riscv.Registers.A0)


def test_immediate_s_inst():
    # S-Type - 12-bits signed immediate
    lb, ub = Signedness.SIGNED.value_range(12)
    a1 = create_ssa_value(riscv.Registers.A1)
    a2 = create_ssa_value(riscv.Registers.A2)

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
    a1 = create_ssa_value(riscv.Registers.A1)

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
        rv32.LiOp(ub, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.LiOp(lb - 1, rd=riscv.Registers.A0)

    rv32.LiOp(ub - 1, rd=riscv.Registers.A0)
    rv32.LiOp(lb, rd=riscv.Registers.A0)


def test_immediate_shift_inst():
    # Shift instructions (SLLI, SRLI, SRAI) - 5-bits immediate
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.SlliOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.SlliOp(a1, -1, rd=riscv.Registers.A0)

    rv32.SlliOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_float_register():
    with pytest.raises(
        VerifyException, match="Invalid register name ft9 for register type riscv.reg."
    ):
        riscv.IntRegisterType.from_name("ft9")
    with pytest.raises(
        VerifyException, match="Invalid register name a0 for register type riscv.freg."
    ):
        riscv.FloatRegisterType.from_name("a0")

    a1 = create_ssa_value(riscv.Registers.A1)
    a2 = create_ssa_value(riscv.Registers.A2)
    with pytest.raises(VerifyException, match="Operation does not verify"):
        riscv.FAddSOp(a1, a2).verify()

    f1 = create_ssa_value(riscv.Registers.FT0)
    f2 = create_ssa_value(riscv.Registers.FT1)
    riscv.FAddSOp(f1, f2).verify()


def test_riscv_parse_immediate_value():
    ctx = Context()
    ctx.load_dialect(riscv.RISCV)

    prog = """riscv.jalr %0, 1.1, !riscv.reg : (!riscv.reg) -> ()"""
    parser = Parser(ctx, prog)
    with pytest.raises(ParseError, match="Expected immediate"):
        parser.parse_operation()


def test_asm_section():
    section = riscv.AssemblySectionOp("section")
    section.verify()


def test_get_constant_value():
    li_op = rv32.LiOp(1)
    li_val = get_constant_value(li_op.rd)
    assert li_val == IntegerAttr.from_int_and_width(1, 32)
    # LiOp implements ConstantLikeInterface so it also has a get_constant_value method:
    constantlike = li_op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(li_op) == IntegerAttr.from_int_and_width(
        1, 32
    )
    zero_op = riscv.GetRegisterOp(riscv.Registers.ZERO)
    zero_val = get_constant_value(zero_op.res)
    assert zero_val == IntegerAttr.from_int_and_width(0, 32)


def test_int_abi_name_by_index():
    assert riscv.IntRegisterType.abi_name_by_index() == {
        0: "zero",
        1: "ra",
        2: "sp",
        3: "gp",
        4: "tp",
        5: "t0",
        6: "t1",
        7: "t2",
        8: "s0",
        9: "s1",
        10: "a0",
        11: "a1",
        12: "a2",
        13: "a3",
        14: "a4",
        15: "a5",
        16: "a6",
        17: "a7",
        18: "s2",
        19: "s3",
        20: "s4",
        21: "s5",
        22: "s6",
        23: "s7",
        24: "s8",
        25: "s9",
        26: "s10",
        27: "s11",
        28: "t3",
        29: "t4",
        30: "t5",
        31: "t6",
    }


def test_int_from_index():
    assert riscv.IntRegisterType.from_index(0) == riscv.Registers.ZERO
    assert riscv.IntRegisterType.from_index(10) == riscv.Registers.A0
    assert riscv.IntRegisterType.from_index(20) == riscv.Registers.S4
    assert riscv.IntRegisterType.from_index(30) == riscv.Registers.T5

    with pytest.raises(KeyError):
        riscv.IntRegisterType.from_index(40)

    assert riscv.IntRegisterType.from_index(
        -10
    ) == riscv.IntRegisterType.infinite_register(9)


def test_float_abi_name_by_index():
    assert riscv.FloatRegisterType.abi_name_by_index() == {
        0: "ft0",
        1: "ft1",
        2: "ft2",
        3: "ft3",
        4: "ft4",
        5: "ft5",
        6: "ft6",
        7: "ft7",
        8: "fs0",
        9: "fs1",
        10: "fa0",
        11: "fa1",
        12: "fa2",
        13: "fa3",
        14: "fa4",
        15: "fa5",
        16: "fa6",
        17: "fa7",
        18: "fs2",
        19: "fs3",
        20: "fs4",
        21: "fs5",
        22: "fs6",
        23: "fs7",
        24: "fs8",
        25: "fs9",
        26: "fs10",
        27: "fs11",
        28: "ft8",
        29: "ft9",
        30: "ft10",
        31: "ft11",
    }


def test_float_from_index():
    assert riscv.FloatRegisterType.from_index(0) == riscv.Registers.FT0
    assert riscv.FloatRegisterType.from_index(10) == riscv.Registers.FA0
    assert riscv.FloatRegisterType.from_index(20) == riscv.Registers.FS4
    assert riscv.FloatRegisterType.from_index(30) == riscv.Registers.FT10

    with pytest.raises(KeyError):
        riscv.FloatRegisterType.from_index(40)

    assert riscv.FloatRegisterType.from_index(
        -10
    ) == riscv.FloatRegisterType.infinite_register(9)
