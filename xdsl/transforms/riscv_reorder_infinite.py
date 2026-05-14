from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.dialects.riscv import RISCVRegisterType
from xdsl.dialects.riscv.ops import ParallelMovOp
from xdsl.ir import Attribute, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def is_infinite(reg: RISCVRegisterType):
    return isinstance(reg.index, IntAttr) and reg.index.data < 0


class ReorderParallelMovPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.ParallelMovOp, rewriter: PatternRewriter):
        # create map from regs to ops that define to regs in this interval
        value_by_reg = defaultdict[Attribute, list[SSAValue]](list)
        cur_op = op
        while cur_op.next_op is not None and not isinstance(
            cur_op.next_op, ParallelMovOp
        ):
            for value in cur_op.results:
                value_by_reg[value.type].append(value)
            cur_op = cur_op.next_op

        # Greedily swap regs to optimal for each in/out pair
        for in_val, out_val in zip(op.inputs, op.outputs, strict=True):
            in_reg = cast(RISCVRegisterType, in_val.type)
            out_reg = out_val.type
            # Only consider swapping if both in and out is infinite
            # and that regs are different
            if in_reg != out_reg and is_infinite(in_reg) and is_infinite(out_reg):
                if in_reg in value_by_reg:
                    # If in_reg is already in interval, we need to swap the regs
                    # (in_reg <-> out_reg in interval)
                    new_out_regs: list[SSAValue] = []
                    for value in value_by_reg[in_reg]:
                        new_out_regs.append(
                            rewriter.replace_value_with_new_type(value, out_reg)
                        )

                    new_in_regs: list[SSAValue] = []
                    for value in value_by_reg[out_reg]:
                        rewriter.replace_value_with_new_type(value, in_reg)
                    # update map
                    value_by_reg[in_reg] = new_in_regs
                    value_by_reg[out_reg] = new_out_regs
                else:
                    # If its not already in the interval, we can just replace out_reg by in_reg
                    new_in_regs: list[SSAValue] = []
                    for value in value_by_reg[out_reg]:
                        rewriter.replace_value_with_new_type(value, in_reg)
                    # update map
                    value_by_reg[in_reg] = new_in_regs


@dataclass(frozen=True)
class RISCVReorderInfinitePass(ModulePass):
    """Reorder infinite registers to optimise for ParallelMovOps."""

    name = "riscv-reorder-infinite"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ReorderParallelMovPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
