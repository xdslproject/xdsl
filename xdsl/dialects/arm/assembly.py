from typing import TypeAlias

from xdsl.dialects.arm.register import ARMRegisterType
from xdsl.ir import SSAValue

AssemblyInstructionArg: TypeAlias = ARMRegisterType | SSAValue


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, ARMRegisterType):
        reg = arg.register_name
        return reg
    else:  # SSAValue
        if isinstance(arg.type, ARMRegisterType):
            reg = arg.type.register_name
            return reg
        else:
            raise ValueError(f"Unexpected argument type {type(arg)}")
