from xdsl.backend.assembly_printer import reg
from xdsl.dialects.arm.registers import IntRegisterType
from xdsl.utils.test_value import create_ssa_value


def test_assembly_arg_str_ARMRegister():
    x0 = IntRegisterType.from_name("x0")
    x0_val = create_ssa_value(x0)
    assert reg(x0_val) == "x0"
