from typing import override

import pytest

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class TestRegister(RegisterType):
    """
    A RISC-V register type.
    """

    name = "test.reg"

    def verify(self) -> None: ...

    @classmethod
    def instruction_set_name(cls) -> str:
        return "TEST"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return {"x0": 0}

    @classmethod
    @override
    def infinite_register_name(cls, index: int) -> str:
        return f"x{index}"


def test_register_type():
    with pytest.raises(AssertionError):
        TestRegister.infinite_register(0)
