from typing import TypeAlias

from xdsl.dialects.arm.register import ARMRegisterType
from xdsl.ir import SSAValue

AssemblyInstructionArg: TypeAlias = ARMRegisterType | SSAValue | str


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, ARMRegisterType):
        reg = arg.register_name
        return reg
    elif isinstance(arg, str):
        return arg
    else:  # SSAValue
        if isinstance(arg.type, ARMRegisterType):
            reg = arg.type.register_name
            return reg
        else:
            raise ValueError(f"Unexpected argument type {type(arg)}")
