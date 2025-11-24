from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import riscv, riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


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
                    rd=riscv.IntRegisterType.a_register(i),
                )
            )

        ops.append(riscv.LiOp(immediate=op.syscall_num, rd=riscv.Registers.A7))

        if op.result is None:
            ops.append(riscv.EcallOp())
            new_results = ()
        else:
            # The result will be stored to a0, move to register that will be used
            ecall = riscv.EcallOp()
            ops.append(ecall)
            gr = riscv.GetRegisterOp(riscv.Registers.A0)
            ops.append(gr)
            res = gr.res

            mv = riscv.MVOp(res, rd=op.result.type)
            ops.append(mv)
            new_results = mv.results

        rewriter.replace_op(op, ops, new_results=new_results)


class InsertExitSyscallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.ReturnOp, rewriter: PatternRewriter):
        parent_op = op.parent_op()
        if (
            not isinstance(parent_op, riscv_func.FuncOp)
            or parent_op.sym_name.data != "main"
        ):
            return

        EXIT = 93
        rewriter.insert_op(riscv_func.SyscallOp(EXIT))


@dataclass(frozen=True)
class LowerRISCVFunc(ModulePass):
    name = "lower-riscv-func"

    insert_exit_syscall: bool = field(default=False)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        if self.insert_exit_syscall:
            PatternRewriteWalker(
                InsertExitSyscallOp(), apply_recursively=False
            ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerSyscallOp(),
                ]
            )
        ).rewrite_module(op)
