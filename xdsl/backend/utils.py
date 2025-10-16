from collections.abc import Callable, Sequence

from xdsl.backend.register_type import RegisterType
from xdsl.builder import Builder
from xdsl.dialects.builtin import (
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, SSAValue


def cast_to_regs(
    values: Sequence[SSAValue],
    register_map: Callable[[Attribute], type[RegisterType]],
    builder: Builder,
) -> list[SSAValue[Attribute]]:
    """
    Return cast operations for operands that don't already have a register type
    and the new list of values that are all guaranteed to have register types.
    """
    registers: list[SSAValue] = []
    for v in values:
        if isinstance(v.type, RegisterType):
            new_value = v
        else:
            cast_op, new_value = UnrealizedConversionCastOp.cast_one(
                v, register_map(v.type).unallocated()
            )
            builder.insert_op(cast_op)
        registers.append(new_value)
    return registers
