from xdsl.dialects.arm.assembly import assembly_arg_str
from xdsl.dialects.arm.attrs import LabelAttr
from xdsl.dialects.arm.register import IntRegisterType


def test_assembly_arg_str_ARMRegister():
    arg = IntRegisterType("x0")
    assert assembly_arg_str(arg) == "x0"


def test_assembly_arg_str_LabelAttr():
    arg = LabelAttr("main")
    assert assembly_arg_str(arg) == "main"
