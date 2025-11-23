from dataclasses import dataclass, field

from xdsl.backend.riscv.lowering.utils import cast_matched_op_results
from xdsl.backend.x86.lowering.helpers import Arch
from xdsl.context import Context
from xdsl.dialects import builtin, scf, x86_scf
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def cast_block_args_to_regs(block: Block, arch: Arch, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to registers and add cast operations just after
    the block entry.
    """

    for arg in block.args:
        rewriter.insert_op(
            cast_op := builtin.UnrealizedConversionCastOp(
                operands=[arg], result_types=[arg.type]
            ),
            InsertPoint.at_start(block),
        )
        new_val = cast_op.results[0]

        new_type = arch.register_type_for_type(arg.type).unallocated()
        arg.replace_by_if(new_val, lambda use: use.operation != cast_op)
        rewriter.replace_value_with_new_type(arg, new_type)


@dataclass
class ScfForLowering(RewritePattern):
    arch: Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        lb, ub, step, *args = self.arch.cast_operands_to_regs(rewriter)
        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        cast_block_args_to_regs(new_region.block, self.arch, rewriter)

        values = tuple(
            self.arch.move_value_to_unallocated(value, value_type, rewriter)
            for value, value_type in zip(args, op.iter_args.types)
        )

        cast_matched_op_results(rewriter)

        new_op = rewriter.insert_op(x86_scf.ForOp(lb, ub, step, values, new_region))
        rewriter.insertion_point = InsertPoint.after(op)

        res_values = tuple(
            builtin.UnrealizedConversionCastOp.cast_one(res_value, res_value_type)
            for res_value, res_value_type in zip(new_op.results, op.results.types)
        )

        for res, (cast_op, result) in zip(
            rewriter.current_operation.results, res_values
        ):
            rewriter.insert_op(cast_op)
            res.replace_by_if(result, lambda use: use.operation is not cast_op)

        rewriter.erase_op(op)


@dataclass
class ScfYieldLowering(RewritePattern):
    arch: Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.YieldOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(
            op, x86_scf.YieldOp(*self.arch.cast_operands_to_regs(rewriter))
        )


@dataclass(frozen=True)
class ConvertScfToX86ScfPass(ModulePass):
    name = "convert-scf-to-x86-scf"

    arch: str = field(default="unknown")

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        arch = Arch.arch_for_name(self.arch)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ScfYieldLowering(arch),
                    ScfForLowering(arch),
                ]
            )
        ).rewrite_module(op)
