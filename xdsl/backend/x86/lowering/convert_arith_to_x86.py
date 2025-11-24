from dataclasses import dataclass

from xdsl.backend.x86.lowering.helpers import Arch
from xdsl.context import Context
from xdsl.dialects import arith, builtin, x86
from xdsl.dialects.builtin import (
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation
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
            immediate=op.value.value.data, destination=x86.registers.UNALLOCATED_GENERAL
        )
        cast_op, _ = UnrealizedConversionCastOp.cast_one(
            mov_op.destination, op.result.type
        )
        rewriter.replace_op(op, [mov_op, cast_op])


X86_OP_BY_ARITH_BINARY_OP = {
    arith.AddiOp: x86.RS_AddOp,
    arith.AddfOp: x86.RS_FAddOp,
    arith.MuliOp: x86.RS_ImulOp,
    arith.MulfOp: x86.RS_FMulOp,
}


@dataclass
class ArithBinaryToX86(RewritePattern):
    arch: Arch

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        new_type = X86_OP_BY_ARITH_BINARY_OP.get(type(op))  # pyright: ignore
        if new_type is None:
            return

        lhs = op.operands[0]

        if isinstance(lhs.type, builtin.ShapedType):
            raise DiagnosticException(
                f"Lowering of {op.name} not implemented for ShapedType"
            )
        rewriter.name_hint = op.results[0].name_hint

        lhs_x86, rhs_x86 = self.arch.cast_operands_to_regs(rewriter)
        moved_rhs = self.arch.move_value_to_unallocated(
            rhs_x86, op.operands[1].type, rewriter
        )
        add_op = new_type(source=lhs_x86, register_in=moved_rhs)
        result_cast_op, _ = UnrealizedConversionCastOp.cast_one(
            add_op.register_out, lhs.type
        )
        rewriter.replace_op(op, [add_op, result_cast_op])


@dataclass(frozen=True)
class ConvertArithToX86Pass(ModulePass):
    name = "convert-arith-to-x86"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        arch = Arch.arch_for_name(None)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ArithBinaryToX86(arch),
                    ArithConstantToX86(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
