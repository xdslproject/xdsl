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

SCALL_EXIT = 93
"""
93 is the number of the `exit` syscall on RISCV.
"""


class LowerRISCVFuncOp(RewritePattern):
    """
    Temporary lowering of only the main function, would like to lower non-main functions
    soon.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.FuncOp, rewriter: PatternRewriter):
        body = op.func_body.block
        # TODO: replace with insert_at_start
        first_op = body.first_op
        assert first_op is not None
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

        rewriter.inline_block_after(op.func_body.block, op)
        rewriter.replace_matched_op(riscv.LabelOp(op.func_name.data))


class LowerRISCVFuncReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.ReturnOp, rewriter: PatternRewriter):
        func_op = op.parent_op()
        if func_op is not None:
            assert isinstance(func_op, riscv_func.FuncOp)
            if func_op.func_name.data == "main":
                # Main does not return, call exit syscall instead
                rewriter.replace_matched_op(riscv_structured.SyscallOp(SCALL_EXIT))
                return

        if op.value is not None:
            rewriter.insert_op_before_matched_op(
                riscv.MVOp(op.value, rd=riscv.Registers.A0)
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
            riscv.JalOp(immediate=op.func_name.data),
        ]

        if op.result is not None:
            get_a0 = riscv.GetRegisterOp(riscv.Registers.A0)
            move_res = riscv.MVOp(get_a0)
            ops.extend((get_a0, move_res))
            new_results = move_res.results
        else:
            new_results = []
        rewriter.replace_matched_op(ops, new_results=new_results)


class LowerRISCVFunc(ModulePass):
    name = "lower-riscv-func"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerRISCVFuncReturnOp()).rewrite_module(op)
        PatternRewriteWalker(LowerRISCVFuncOp()).rewrite_module(op)
        PatternRewriteWalker(LowerRISCVCallOp()).rewrite_module(op)
