from xdsl.backend.assembly_printer import reg
from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.test_value import create_ssa_value


@irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return {"x0": 0, "x1": 1, "a0": 0, "a1": 1}

    @classmethod
    def infinite_register_prefix(cls):
        return "y"


def test_reg():
    x0_val = create_ssa_value(TestRegister.from_name("x0"))
    x1_val = create_ssa_value(TestRegister.from_name("x1"))
    assert reg(x0_val) == "x0"
    assert reg(x1_val) == "x1"
