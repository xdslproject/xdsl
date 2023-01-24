import pytest

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.riscv import BEQOp, BGTOp, DirectiveOp, JALOp, LUIOp, LabelOp, MVOp, RegisterAttr, LabelAttr, LBOp, JALROp, SBOp, SCALLOp, SLLOp


def test_registerattr():
    reg1 = RegisterAttr.from_index(13)
    assert reg1.data.index == 13
    assert reg1.data.get_abi_name() == "a3"

    reg2 = RegisterAttr.from_name("a3")
    assert reg2.data.get_abi_name() == "a3"
    assert reg2.data.index == 13

    reg3 = RegisterAttr.from_register(reg1.data)
    assert reg3.data.index == 13
    assert reg3.data.get_abi_name() == "a3"


def test_labelattr():
    str = "amazing_string"
    label = LabelAttr.from_str(str)
    assert label.data == str


def test_riscv1rd1rs1immoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = LBOp.get(reg0, reg1, 42)
    op2 = LBOp.get(reg0, reg1, val)

    assert op1.rd == reg0
    assert op1.rs1 == reg1
    assert op1.immediate == val

    assert op2.rd == reg0
    assert op2.rs1 == reg1
    assert op2.immediate == val


def test_riscv1rd1rs1offoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = JALROp.get(reg0, reg1, 42)
    op2 = JALROp.get(reg0, reg1, val)

    assert op1.rd == reg0
    assert op1.rs1 == reg1
    assert op1.offset == val

    assert op2.rd == reg0
    assert op2.rs1 == reg1
    assert op2.offset == val

    str = LabelAttr.from_str("amazing")

    op1 = JALROp.get(reg0, reg1, "amazing")
    op2 = JALROp.get(reg0, reg1, str)

    assert op1.rd == reg0
    assert op1.rs1 == reg1
    assert op1.offset == str

    assert op2.rd == reg0
    assert op2.rs1 == reg1
    assert op2.offset == str


def test_riscv2rs1immoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = SBOp.get(reg0, reg1, 42)
    op2 = SBOp.get(reg0, reg1, val)

    assert op1.rs1 == reg0
    assert op1.rs2 == reg1
    assert op1.immediate == val

    assert op2.rs1 == reg0
    assert op2.rs2 == reg1
    assert op2.immediate == val


def test_riscv2rs1offoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = BEQOp.get(reg0, reg1, 42)
    op2 = BEQOp.get(reg0, reg1, val)

    assert op1.rs1 == reg0
    assert op1.rs2 == reg1
    assert op1.offset == val

    assert op2.rs1 == reg0
    assert op2.rs2 == reg1
    assert op2.offset == val

    str = LabelAttr.from_str("amazing")

    op1 = BEQOp.get(reg0, reg1, "amazing")
    op2 = BEQOp.get(reg0, reg1, str)

    assert op1.rs1 == reg0
    assert op1.rs2 == reg1
    assert op1.offset == str

    assert op2.rs1 == reg0
    assert op2.rs2 == reg1
    assert op2.offset == str


def test_riscv1rd2rsoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)
    reg2 = RegisterAttr.from_index(2)

    op = SLLOp.get(reg0, reg1, reg2)

    assert op.rd == reg0
    assert op.rs1 == reg1
    assert op.rs2 == reg2


def test_riscv1rs1rt1offoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = BGTOp.get(reg0, reg1, 42)
    op2 = BGTOp.get(reg0, reg1, val)

    assert op1.rs == reg0
    assert op1.rt == reg1
    assert op1.offset == val

    assert op2.rs == reg0
    assert op2.rt == reg1
    assert op2.offset == val

    str = LabelAttr.from_str("amazing")

    op1 = BGTOp.get(reg0, reg1, "amazing")
    op2 = BGTOp.get(reg0, reg1, str)

    assert op1.rs == reg0
    assert op1.rt == reg1
    assert op1.offset == str

    assert op2.rs == reg0
    assert op2.rt == reg1
    assert op2.offset == str


def test_riscv1rd1immoperation():
    reg0 = RegisterAttr.from_index(0)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = LUIOp.get(reg0, 42)
    op2 = LUIOp.get(reg0, val)

    assert op1.rd == reg0
    assert op1.immediate == val

    assert op2.rd == reg0
    assert op2.immediate == val


def test_riscv1rd1offoperation():
    reg0 = RegisterAttr.from_index(0)
    val = IntegerAttr.from_int_and_width(42, 64)

    op1 = JALOp.get(reg0, 42)
    op2 = JALOp.get(reg0, val)

    assert op1.rd == reg0
    assert op1.offset == val

    assert op2.rd == reg0
    assert op2.offset == val

    str = LabelAttr.from_str("amazing")

    op1 = JALOp.get(reg0, "amazing")
    op2 = JALOp.get(reg0, str)

    assert op1.rd == reg0
    assert op1.offset == str

    assert op2.rd == reg0
    assert op2.offset == str


def test_riscv1rd1rsoperation():
    reg0 = RegisterAttr.from_index(0)
    reg1 = RegisterAttr.from_index(1)

    op = MVOp.get(reg0, reg1)

    assert op.rd == reg0
    assert op.rs == reg1


def test_riscvnoparamsoperation():
    SCALLOp.get()


def test_labelop():
    label = LabelAttr.from_str("amazing")

    op1 = LabelOp.get(label)
    op2 = LabelOp.get("amazing")

    assert op1.label == label
    assert op2.label == label


def test_directiveop():
    directive = StringAttr.from_str("directive")
    value = StringAttr.from_str("value")

    op1 = DirectiveOp.get(directive, value)
    op2 = DirectiveOp.get("directive", value)
    op3 = DirectiveOp.get(directive, "value")
    op4 = DirectiveOp.get("directive", "value")

    assert op1.directive == directive
    assert op1.value == value
    assert op2.directive == directive
    assert op2.value == value
    assert op3.directive == directive
    assert op3.value == value
    assert op4.directive == directive
    assert op4.value == value
