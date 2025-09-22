import re

import pytest

from xdsl.backend.assembly_printer import reg
from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import i32
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.exceptions import DiagnosticException
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

    assert reg(x0_val, x0_val) == "x0"

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Expected types to be same, got ['!test.reg<x0>', '!test.reg<x1>']"
        ),
    ):
        reg(x0_val, x1_val)

    i32_val = create_ssa_value(i32)

    with pytest.raises(DiagnosticException, match="Expected register type, got i32"):
        reg(i32_val)
