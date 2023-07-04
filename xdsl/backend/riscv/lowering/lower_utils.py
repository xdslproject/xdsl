from typing import Sequence

from xdsl.dialects import riscv
from xdsl.dialects import builtin
from xdsl.dialects.builtin import AnyFloat
from xdsl.ir.core import Attribute, OpResult, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_values_to_registers(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[OpResult]:
    if not operands:
        return []
    types = [
        riscv.FloatRegisterType(riscv.Register())
        if isinstance(op.typ, AnyFloat)
        else riscv.RegisterType(riscv.Register())
        for op in operands
    ]
    cast = builtin.UnrealizedConversionCastOp.get(operands, types)
    rewriter.insert_op_before_matched_op(cast)
    return cast.results


def get_type_size(t: Attribute) -> int:
    if isinstance(t, builtin.Float32Type):
        return 4
    elif isinstance(t, builtin.IntegerType) and t.width == 32:
        return 4
    raise NotImplementedError(f"Type {t} is not supported")
