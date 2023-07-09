from xdsl.dialects import memref, printf, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir.core import Attribute, MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class LowerPrintOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        assert op.format_str.data == "{}"
        param = op.format_vals[0]
        memref_typ = param.type
        assert isa(memref_typ, memref.MemRefType[Attribute])
        shape = memref_typ.get_shape()
        assert len(shape) == 2
        rewriter.replace_matched_op(
            [
                rows := riscv.LiOp(shape[0]),
                cols := riscv.LiOp(shape[1]),
                input := UnrealizedConversionCastOp.get(
                    (param,), (riscv.RegisterType(riscv.Register()),)
                ),
                riscv.CustomAssemblyInstructionOp(
                    "tensor.print2d", (input.results[0], rows.rd, cols.rd), ()
                ),
            ]
        )


class LowerPrintfRiscvPass(ModulePass):
    name = "lower-printf-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerPrintOp()).rewrite_module(op)
