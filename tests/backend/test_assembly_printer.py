from xdsl.backend.assembly_printer import RegisterNameSpec, reg
from xdsl.backend.register_type import NamedRegisterType, RegisterType
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.test_value import create_ssa_value


@irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"


@irdl_attr_definition
class TestNamedRegister(NamedRegisterType):
    name = "test.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return {"x0": 0, "x1": 1, "a0": 0, "a1": 1}

    @classmethod
    def infinite_register_prefix(cls):
        return "y"


def test_reg():
    x0_val = create_ssa_value(TestNamedRegister.from_name("x0"))
    x1_val = create_ssa_value(TestNamedRegister.from_name("x1"))
    a0_val = create_ssa_value(TestNamedRegister.from_name("a0"))
    a1_val = create_ssa_value(TestNamedRegister.from_name("a1"))
    assert reg(x0_val) == "a0"
    assert reg(x1_val) == "a1"
    assert reg(a0_val) == "a0"
    assert reg(a1_val) == "a1"


def test_register_name_spec():
    class TestSpec(RegisterNameSpec):
        def get_register_name(self, index: int) -> str:
            return f"hello{index}"

    x0_val = create_ssa_value(TestRegister.from_index(0))
    x1_val = create_ssa_value(TestRegister.from_index(1))

    x0_named_val = create_ssa_value(TestNamedRegister.from_name("x0"))
    x1_named_val = create_ssa_value(TestNamedRegister.from_name("x1"))

    assert reg(x0_val, TestSpec()) == "hello0"
    assert reg(x1_val, TestSpec()) == "hello1"
    assert reg(x0_named_val, TestSpec()) == "hello0"
    assert reg(x1_named_val, TestSpec()) == "hello1"
