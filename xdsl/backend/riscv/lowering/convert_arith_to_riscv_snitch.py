from dataclasses import dataclass
from math import prod
from typing import Any, cast

from xdsl.context import MLContext
from xdsl.dialects import arith, riscv, riscv_snitch
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    ModuleOp,
    UnrealizedConversionCastOp,
    VectorType,
)
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

_FLOAT_REGISTER_TYPE = riscv.FloatRegisterType.unallocated()


@dataclass
class LowerBinaryFloatVectorOp(RewritePattern):
    arith_op_cls: type[arith.FloatingPointLikeBinaryOperation]
    riscv_d_op_cls: type[riscv.RdRsRsFloatOperationWithFastMath]
    riscv_snitch_v_f32_op_cls: type[riscv.RdRsRsFloatOperationWithFastMath]
    riscv_snitch_v_f16_op_cls: type[riscv.RdRsRsFloatOperationWithFastMath]

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        operand_type = op.result.type
        if not isinstance(operand_type, VectorType):
            return
        shape = operand_type.shape
        count = prod(dim.data for dim in shape.data)

        operand_type = cast(VectorType[Any], operand_type)
        scalar_type = operand_type.element_type

        lhs = UnrealizedConversionCastOp.get((op.lhs,), (_FLOAT_REGISTER_TYPE,))
        rhs = UnrealizedConversionCastOp.get((op.rhs,), (_FLOAT_REGISTER_TYPE,))

        match scalar_type:
            case Float64Type():
                if count != 1:
                    return
                cls = self.riscv_d_op_cls
            case Float32Type():
                if count != 2:
                    return
                cls = self.riscv_snitch_v_f32_op_cls
            case Float16Type():
                if count != 4:
                    return
                cls = self.riscv_snitch_v_f16_op_cls
            case _:
                assert False, f"Unexpected float type {op.lhs.type}"

        rv_flags = riscv.FastMathFlagsAttr("none")
        if op.fastmath is not None:
            rv_flags = riscv.FastMathFlagsAttr(op.fastmath.data)

        new_op = cls(lhs, rhs, rd=_FLOAT_REGISTER_TYPE, fastmath=rv_flags)
        cast_op = UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,))

        rewriter.replace_matched_op((lhs, rhs, new_op, cast_op))


lower_arith_addf = LowerBinaryFloatVectorOp(
    arith.Addf, riscv.FAddDOp, riscv_snitch.VFAddSOp, riscv_snitch.VFAddHOp
)


class ConvertArithToRiscvSnitchPass(ModulePass):
    name = "convert-arith-to-riscv-snitch"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    lower_arith_addf,
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
