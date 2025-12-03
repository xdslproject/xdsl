from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException


class ParallelMovPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.ParallelMovOp, rewriter: PatternRewriter):
        input_types = cast(Sequence[riscv.RISCVRegisterType], op.inputs.types)
        output_types = cast(Sequence[riscv.RISCVRegisterType], op.outputs.types)

        if not (
            all(i.is_allocated for i in input_types)
            and all(i.is_allocated for i in output_types)
        ):
            raise PassFailedException("All registers must be allocated")

        num_operands = len(op.operands)

        new_ops: list[Operation] = []
        results: list[SSAValue | None] = [None] * num_operands

        dst_to_src: dict[riscv.RegisterType, SSAValue] = {}

        # registers which are outputs but not inputs
        end_regs: set[riscv.RegisterType, SSAValue] = set(op.outputs.types)

        for idx, src, dst in zip(
            range(num_operands), op.inputs, op.outputs, strict=True
        ):
            end_regs.discard(src.type)

            if src.type == dst.type:
                results[idx] = src
            else:
                dst_to_src[dst.type] = src

        for dst in end_regs:
            while dst in dst_to_src:
                src = dst_to_src[dst]
                new_ops.append(riscv.MVOp(src, rd=dst))
                results[op.outputs.types.index(dst)] = new_ops[-1].results[0]
                dst = src.type

        rewriter.replace_matched_op(new_ops, results)


@dataclass(frozen=True)
class RISCVLowerParallelMovPass(ModulePass):
    """Lowers ParallelMovOp in a module into separate moves."""

    name = "riscv-lower-parallel-mov"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ParallelMovPattern()).rewrite_module(op)
