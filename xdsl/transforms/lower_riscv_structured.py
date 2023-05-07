from typing import cast
from xdsl.dialects.builtin import ModuleOp

from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.dialects import riscv, riscv_structured
from xdsl.transforms.dead_code_elimination import dce


class LowerSyscallOp(RewritePattern):
    """
    Lower SSA version of syscall, storing the optional result to a0.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_structured.SyscallOp, rewriter: PatternRewriter
    ):
        ops: list[Operation] = []

        for i, arg in enumerate(op.args):
            ops.append(
                riscv.MVOp(
                    arg,
                    rd=riscv.Register(f"a{i}"),
                )
            )

        ops.append(riscv.LiOp(immediate=op.syscall_num, rd=riscv.Registers.A7))

        if op.result is None:
            ops.append(riscv.EcallOp())
            new_results = []
        else:
            # The result will be stored to a0, move to register that will be used
            ecall = riscv.EcallOp()
            ops.append(ecall)
            gr = riscv.GetRegisterOp(riscv.Registers.A0)
            ops.append(gr)
            res = gr.res

            mv = riscv.MVOp(res, rd=cast(riscv.RegisterType, op.result.typ))
            ops.append(mv)
            new_results = mv.results

        rewriter.replace_matched_op(ops, new_results=new_results)


class LowerRISCVStructured(ModulePass):
    name = "lower-riscv-structured"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerSyscallOp()).rewrite_module(op)
        dce(op)
