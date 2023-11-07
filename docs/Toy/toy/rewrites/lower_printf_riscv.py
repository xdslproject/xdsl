from xdsl.dialects import memref, printf, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import Attribute, MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class LowerPrintOp(RewritePattern):
    """
    Rewrites printf.PrintFormatOp to Toy accelerator custom instruction. Currently only
    supports 1d and 2d memref printing.

    This is a temporary pass, until we have the ability to print in native riscv.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        assert op.format_str.data == "{}"
        param = op.format_vals[0]
        memref_typ = param.type
        assert isa(memref_typ, memref.MemRefType[Attribute])
        shape = memref_typ.get_shape()
        match shape:
            case [element_count]:
                rewriter.replace_matched_op(
                    [
                        rows := riscv.LiOp(element_count),
                        input := UnrealizedConversionCastOp.get(
                            (param,), (riscv.IntRegisterType.unallocated(),)
                        ),
                        riscv.CustomAssemblyInstructionOp(
                            "tensor.print1d", (input.results[0], rows.rd), ()
                        ),
                    ]
                )
            case [row_count, column_count]:
                rewriter.replace_matched_op(
                    [
                        rows := riscv.LiOp(row_count),
                        cols := riscv.LiOp(column_count),
                        input := UnrealizedConversionCastOp.get(
                            (param,), (riscv.IntRegisterType.unallocated(),)
                        ),
                        riscv.CustomAssemblyInstructionOp(
                            "tensor.print2d", (input.results[0], rows.rd, cols.rd), ()
                        ),
                    ]
                )
            case _:
                raise NotImplementedError(
                    f"Cannot print memref with length {len(shape)}"
                )


class LowerPrintfRiscvPass(ModulePass):
    name = "lower-printf-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerPrintOp()).rewrite_module(op)
