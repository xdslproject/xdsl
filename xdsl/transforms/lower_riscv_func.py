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
from xdsl.dialects import riscv, riscv_func
from xdsl.transforms.dead_code_elimination import dce


class LowerSyscallOp(RewritePattern):
    """
    Lower SSA version of syscall, storing the optional result to a0.

    Different platforms have different calling conventions. This lowering assumes that
    the inputs are stored in a0-a6, and the opcode is stored to a7. Upon return, the
    a0 contains the result value. This is not the case for some kernels.

    In the future, this pass should take the compilation target as a parameter to guide
    the rewrites.

    Issue tracking this: https://github.com/xdslproject/xdsl/issues/952
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.SyscallOp, rewriter: PatternRewriter):
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


class LowerRISCVFunc(ModulePass):
    name = "lower-riscv-func"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerSyscallOp()).rewrite_module(op)
        dce(op)
