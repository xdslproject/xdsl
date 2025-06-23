from collections.abc import Callable, Sequence

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import (
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_values_to_regs(
    values: Sequence[SSAValue],
    rewriter: PatternRewriter,
    register_map: Callable[[Attribute], RegisterType],
) -> Sequence[SSAValue[RegisterType]]:
    registers: list[SSAValue[RegisterType]] = []
    for v in values:
        cast_op, cast_value = UnrealizedConversionCastOp.cast_one(
            v, register_map(v.type)
        )
        rewriter.insert_op_before_matched_op(cast_op)
        registers.append(cast_value)
    return registers
