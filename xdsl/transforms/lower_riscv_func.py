from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects import riscv, riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Operation, OpResult
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

            mv = riscv.MVOp(res, rd=cast(riscv.RegisterType, op.result.type))
            ops.append(mv)
            new_results = mv.results

        rewriter.replace_matched_op(ops, new_results=new_results)


class LowerRISCVFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.FuncOp, rewriter: PatternRewriter):
        # TODO CFToRISCV lowering
        # body = op.func_body.block
        # first_op = body.first_op
        # assert first_op is not None
        for body in op.func_body.blocks:
            first_op = body.first_op
            # assert first_op is not None
            if first_op is None:
                pass
            else:
                while len(body.args):
                    # arguments are passed to riscv functions via a0, a1, ...
                    # replace arguments with `GetRegisterOp`s
                    index = len(body.args) - 1
                    last_arg = body.args[-1]
                    get_reg_op = riscv.GetRegisterOp(riscv.Register(f"a{index}"))
                    last_arg.replace_by(get_reg_op.res)
                    rewriter.insert_op_before(get_reg_op, first_op)
                    first_op = get_reg_op
                    rewriter.erase_block_argument(last_arg)

        label_body = rewriter.move_region_contents_to_new_regions(op.func_body)

        rewriter.replace_matched_op(riscv.LabelOp(op.sym_name.data, region=label_body))


class InsertExitSyscallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data != "main":
            return
        return_op = op.func_body.block.last_op
        assert isinstance(return_op, riscv_func.ReturnOp)

        EXIT = 93
        rewriter.insert_op_before(riscv_func.SyscallOp(EXIT), return_op)


class LowerRISCVFuncReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.ReturnOp, rewriter: PatternRewriter):
        for i, value in enumerate(op.values):
            rewriter.insert_op_before_matched_op(
                riscv.MVOp(value, rd=riscv.Register(f"a{i}"))
            )
        rewriter.replace_matched_op(riscv.ReturnOp())


class LowerRISCVCallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.CallOp, rewriter: PatternRewriter):
        for i, arg in enumerate(op.operands):
            # Load arguments into a0...
            rewriter.insert_op_before_matched_op(
                riscv.MVOp(arg, rd=riscv.Register(f"a{i}"))
            )

        ops: list[Operation] = [
            riscv.JalOp(op.callee.data),
        ]
        new_results: list[OpResult] = []

        for i in range(len(op.results)):
            get_reg = riscv.GetRegisterOp(riscv.Register(f"a{i}"))
            move_res = riscv.MVOp(get_reg)
            ops.extend((get_reg, move_res))
            new_results.append(move_res.rd)

        rewriter.replace_matched_op(ops, new_results=new_results)


@dataclass
class LowerRISCVFunc(ModulePass):
    name = "lower-riscv-func"

    insert_exit_syscall: bool = field(default=False)

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        if self.insert_exit_syscall:
            PatternRewriteWalker(
                InsertExitSyscallOp(), apply_recursively=False
            ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerRISCVFuncReturnOp(),
                    LowerRISCVFuncOp(),
                    LowerRISCVCallOp(),
                    LowerSyscallOp(),
                ]
            )
        ).rewrite_module(op)
