from xdsl.dialects import builtin, riscv
from xdsl.ir.core import OpResult, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_value_to_int_register(
    operand: SSAValue, rewriter: PatternRewriter
) -> OpResult:
    types = (riscv.IntRegisterType.unallocated(),)
    cast = builtin.UnrealizedConversionCastOp.get((operand,), types)
    rewriter.insert_op_before_matched_op(cast)
    return cast.results[0]


def cast_value_to_float_register(
    operand: SSAValue, rewriter: PatternRewriter
) -> OpResult:
    types = (riscv.FloatRegisterType.unallocated(),)
    cast = builtin.UnrealizedConversionCastOp.get((operand,), types)
    rewriter.insert_op_before_matched_op(cast)
    return cast.results[0]
