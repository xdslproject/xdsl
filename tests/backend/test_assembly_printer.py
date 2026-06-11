from xdsl.backend.assembly_printer import RegisterNameSpec, reg
from xdsl.dialects.test import TestRegisterType
from xdsl.utils.test_value import create_ssa_value


def test_reg():
    x0_val = create_ssa_value(TestRegisterType.from_name("x0"))
    x1_val = create_ssa_value(TestRegisterType.from_name("x1"))
    assert reg(x0_val) == "x0"
    assert reg(x1_val) == "x1"


def test_register_name_spec():
    class TestSpec(RegisterNameSpec):
        def get_register_name(self, index: int) -> str:
            return f"hello{index}"

    x0_val = create_ssa_value(TestRegisterType.from_name("x0"))
    x1_val = create_ssa_value(TestRegisterType.from_name("x1"))
    assert reg(x0_val, TestSpec()) == "hello0"
    assert reg(x1_val, TestSpec()) == "hello1"
