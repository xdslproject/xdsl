import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"

    def verify(self) -> None: ...

    @classmethod
    def instruction_set_name(cls) -> str:
        return "TEST"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return {"x0": 0}

    @classmethod
    def infinite_register_prefix(cls):
        return "x"


def test_register_clashes():
    with pytest.raises(
        AssertionError,
        match="Invalid 'infinite' register name: x0 clashes with finite register set",
    ):
        TestRegister.infinite_register(0)
