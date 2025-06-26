from dataclasses import dataclass

from xdsl.backend.x86.lowering.helpers import cast_operands_to_regs
from xdsl.context import Context
from xdsl.dialects import arith, builtin, x86
from xdsl.dialects.builtin import (
    IntegerAttr,
    UnrealizedConversionCastOp,
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
from xdsl.utils.hints import isa


@dataclass
class ArithConstantToX86(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr):
            raise DiagnosticException(
                "Lowering of arith.constant is only implemented for integers"
            )
        mov_op = x86.DI_MovOp(
            immediate=op.value.value.data, destination=x86.register.UNALLOCATED_GENERAL
        )
        cast_op, _ = UnrealizedConversionCastOp.cast_one(
            mov_op.destination, op.result.type
        )
        rewriter.replace_matched_op([mov_op, cast_op])


@dataclass
class ArithAddiToX86(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, builtin.ShapedType):
            raise DiagnosticException(
                "Lowering of arith.addi not implemented for ShapedType"
            )
        lhs_x86, rhs_x86 = cast_operands_to_regs(rewriter=rewriter)
        rhs_copy_op = x86.DS_MovOp(
            source=rhs_x86, destination=x86.register.UNALLOCATED_GENERAL
        )
        add_op = x86.RS_AddOp(source=lhs_x86, register_in=rhs_copy_op.destination)
        result_cast_op, _ = UnrealizedConversionCastOp.cast_one(
            add_op.register_in, op.lhs.type
        )
        rewriter.replace_matched_op([rhs_copy_op, add_op, result_cast_op])


@dataclass
class ArithMuliToX86(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MuliOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, builtin.ShapedType):
            raise DiagnosticException(
                "Lowering of arith.muli not implemented for ShapedType"
            )
        lhs_x86, rhs_x86 = cast_operands_to_regs(rewriter=rewriter)
        rhs_copy_op = x86.DS_MovOp(
            source=rhs_x86, destination=x86.register.UNALLOCATED_GENERAL
        )
        add_op = x86.RS_ImulOp(source=lhs_x86, register_in=rhs_copy_op.destination)
        result_cast_op, _ = UnrealizedConversionCastOp.cast_one(
            add_op.register_in, op.lhs.type
        )
        rewriter.replace_matched_op([rhs_copy_op, add_op, result_cast_op])


@dataclass(frozen=True)
class ConvertArithToX86Pass(ModulePass):
    name = "convert-arith-to-x86"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ArithAddiToX86(),
                    ArithMuliToX86(),
                    ArithConstantToX86(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
