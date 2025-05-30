from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, vector, x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    UnrealizedConversionCastOp,
    VectorType,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException

from .helpers import vector_type_to_register_type


@dataclass
class VectorBroadcastToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.BroadcastOp, rewriter: PatternRewriter):
        # Get the register to be broadcasted
        source_cast_op, source_x86 = UnrealizedConversionCastOp.cast_one(
            op.source, x86.register.UNALLOCATED_GENERAL
        )
        # Actually broadcast the register
        element_type = op.source.type
        assert isinstance(element_type, FixedBitwidthType)
        element_size = element_type.bitwidth
        match element_size:
            case 16:
                raise DiagnosticException(
                    "Half-precision vector broadcast is not implemented yet."
                )
            case 32:
                broadcast = x86.ops.DS_VpbroadcastdOp
            case 64:
                broadcast = x86.ops.DS_VpbroadcastqOp
            case _:
                raise DiagnosticException(
                    "Float precision must be half, single or double."
                )
        broadcast_op = broadcast(
            source=source_x86,
            destination=vector_type_to_register_type(op.vector.type, self.arch),
        )
        # Get back the abstract vector
        dest_cast_op, _ = UnrealizedConversionCastOp.cast_one(
            broadcast_op.destination, op.vector.type
        )

        rewriter.replace_matched_op([source_cast_op, broadcast_op, dest_cast_op])


@dataclass
class VectorFMAToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.FMAOp, rewriter: PatternRewriter):
        vect_type = cast(VectorType, op.acc.type)
        x86_vect_type = vector_type_to_register_type(vect_type, self.arch)
        # Pointer casts
        lhs_cast_op, lhs_new = UnrealizedConversionCastOp.cast_one(
            op.lhs, x86_vect_type
        )
        rhs_cast_op, rhs_new = UnrealizedConversionCastOp.cast_one(
            op.rhs, x86_vect_type
        )
        acc_cast_op, acc_new = UnrealizedConversionCastOp.cast_one(
            op.acc, x86_vect_type
        )
        # Instruction selection
        element_size = cast(FixedBitwidthType, vect_type.get_element_type()).bitwidth
        match element_size:
            case 16:
                raise DiagnosticException(
                    "Half-precision vector load is not implemented yet."
                )
            case 32:
                fma = x86.ops.RSS_Vfmadd231psOp
            case 64:
                fma = x86.ops.RSS_Vfmadd231pdOp
            case _:
                raise DiagnosticException(
                    "Float precision must be half, single or double."
                )
        fma_op = fma(acc_new, lhs_new, rhs_new)

        res_cast_op = UnrealizedConversionCastOp.get(
            (fma_op.register_out,), (vect_type,)
        )
        rewriter.replace_matched_op(
            [lhs_cast_op, rhs_cast_op, acc_cast_op, fma_op, res_cast_op]
        )


@dataclass(frozen=True)
class ConvertVectorToX86Pass(ModulePass):
    name = "convert-vector-to-x86"

    arch: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    VectorFMAToX86(self.arch),
                    VectorBroadcastToX86(self.arch),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
