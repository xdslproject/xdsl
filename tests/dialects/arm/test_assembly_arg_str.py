from xdsl.dialects.arm.register import IntRegisterType


def test_assembly_arg_str_ARMRegister():
    assert IntRegisterType.from_name("x0").assembly_str() == "x0"
