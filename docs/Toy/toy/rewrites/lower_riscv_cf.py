from typing import Sequence

from xdsl.dialects import arith, builtin, cf, riscv, riscv_cf, riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import MLContext, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def cast_values_to_registers(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[OpResult]:
    if not operands:
        return []
    types = [riscv.RegisterType(riscv.Register()) for _ in range(len(operands))]
    cast = builtin.UnrealizedConversionCastOp.get(operands, types)
    rewriter.insert_op_before_matched_op(cast)
    return cast.results


def copy_registers(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[OpResult]:
    if not operands:
        return []
    results: list[OpResult] = []
    for operand in operands:
        mv = riscv.MVOp(operand)
        rewriter.insert_op_before_matched_op(mv)
        results.append(mv.rd)
    return results


class AddLabelsToBlocksInFunction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.FuncOp, rewriter: PatternRewriter):
        if len(op.func_body.blocks) < 2:
            return

        func_name = op.sym_name.data

        for i, block in enumerate(op.func_body.blocks):
            first_op = block.first_op
            if not isinstance(first_op, riscv.LabelOp):
                rewriter.insert_op_at_start(riscv.LabelOp(f"{func_name}.bb{i}"), block)


class LowerBranchOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.Branch, rewriter: PatternRewriter):
        # block_args_to_registers(op.successor, rewriter)

        new_operands = cast_values_to_registers(op.arguments, rewriter)
        new_operands = copy_registers(new_operands, rewriter)

        rewriter.replace_matched_op(riscv_cf.JOp(new_operands, op.successor))


class LowerConditionalBranchOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter):
        # Expect that the then block immediately follows the op, which is usually the
        # case. The branch operations in riscv.cf require the else block to be immediately
        # following the current block. They also require that the first op of the then
        # block is a label.

        # block_args_to_registers(op.then_block, rewriter)
        # block_args_to_registers(op.else_block, rewriter)

        new_then_arguments = cast_values_to_registers(op.else_arguments, rewriter)
        new_then_arguments = copy_registers(new_then_arguments, rewriter)
        new_else_arguments = cast_values_to_registers(op.then_arguments, rewriter)
        new_else_arguments = copy_registers(new_else_arguments, rewriter)

        then_block, else_block = op.else_block, op.then_block

        then_block_first_op = then_block.first_op
        assert isinstance(
            then_block_first_op, riscv.LabelOp
        ), f"Handle non function later {then_block_first_op}"

        if isinstance(op.cond, OpResult) and isinstance(op.cond.op, arith.Cmpi):
            cmpi = op.cond.op
            lhs, rhs = cast_values_to_registers([cmpi.lhs, cmpi.rhs], rewriter)
            match cmpi.predicate.value.data:
                case 0:  # eq
                    rewriter.replace_matched_op(
                        riscv_cf.BeqOp(
                            lhs,
                            rhs,
                            new_then_arguments,
                            new_else_arguments,
                            then_block,
                            else_block,
                        )
                    )
                case 1:  # ne
                    rewriter.replace_matched_op(
                        riscv_cf.BneOp(
                            lhs,
                            rhs,
                            new_then_arguments,
                            new_else_arguments,
                            then_block,
                            else_block,
                        )
                    )
                case 2:  # slt
                    # Opposite of slt is Bge
                    rewriter.replace_matched_op(
                        riscv_cf.BgeOp(
                            lhs,
                            rhs,
                            new_then_arguments,
                            new_else_arguments,
                            then_block,
                            else_block,
                        )
                    )
                case 3:  # sle
                    assert False, "unhandled"
                    # No sle, flip arguments and use ge instead
                    rewriter.replace_matched_op(
                        riscv_cf.BgeOp(
                            rhs,
                            lhs,
                            new_else_arguments,
                            new_then_arguments,
                            else_block,
                            then_block,
                        )
                    )
                case 4:  # sgt
                    assert False, "unhandled"

                    # No sgt, flip arguments and use lt instead
                    rewriter.replace_matched_op(
                        riscv_cf.BltOp(
                            rhs,
                            lhs,
                            new_else_arguments,
                            new_then_arguments,
                            else_block,
                            then_block,
                        )
                    )
                case 5:  # sge
                    rewriter.replace_matched_op(
                        riscv_cf.BltOp(
                            lhs,
                            rhs,
                            new_then_arguments,
                            new_else_arguments,
                            then_block,
                            else_block,
                        )
                    )
                case 6:  # ult
                    rewriter.replace_matched_op(
                        riscv_cf.BgeuOp(
                            lhs,
                            rhs,
                            new_then_arguments,
                            new_else_arguments,
                            then_block,
                            else_block,
                        )
                    )
                case 7:  # ule
                    assert False, "unhandled"

                    # No ule, flip arguments and use geu instead
                    rewriter.replace_matched_op(
                        riscv_cf.BgeuOp(
                            rhs,
                            lhs,
                            new_else_arguments,
                            new_then_arguments,
                            else_block,
                            then_block,
                        )
                    )
                case 8:  # ugt
                    assert False, "unhandled"

                    # No ugt, flip arguments and use ltu instead
                    rewriter.replace_matched_op(
                        riscv_cf.BltuOp(
                            rhs,
                            lhs,
                            new_else_arguments,
                            new_then_arguments,
                            else_block,
                            then_block,
                        )
                    )
                case 9:  # uge
                    rewriter.replace_matched_op(
                        riscv_cf.BltuOp(
                            lhs,
                            rhs,
                            new_then_arguments,
                            new_else_arguments,
                            then_block,
                            else_block,
                        )
                    )
                case _:
                    assert False, f"Unexpected comparison predicate {cmpi.predicate}"
            return

        assert False, f"unhandled {op}"


class LowerCfRiscvCfPass(ModulePass):
    name = "lower-cf-riscv-cf"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(AddLabelsToBlocksInFunction()).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerBranchOp(),
                    LowerConditionalBranchOp(),
                ]
            )
        ).rewrite_module(op)
