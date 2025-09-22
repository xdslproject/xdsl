from xdsl.backend.register_type import RegisterType
from xdsl.ir import SSAValue


def reg(value: SSAValue) -> str:
    """
    A wrapper around SSAValue to be printed in assembly.
    Only valid if the type of the value is a RegisterType.
    """

    assert isinstance(value.type, RegisterType)
    return value.type.register_name.data


def square_brackets_reg(value: SSAValue):
    """
    A wrapper around SSAValue to be printed in assembly.
    Only valid if the type of the value is a RegisterType.
    This class handles the case where a register contains a pointer reference,
    and therefore should be printed within square brackets.
    """
    assert isinstance(value.type, RegisterType)
    return f"[{value.type.register_name.data}]"
