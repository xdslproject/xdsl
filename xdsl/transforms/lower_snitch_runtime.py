from abc import ABC
from typing import cast

from xdsl.backend.riscv.lowering.utils import (
    a_regs,
    move_to_a_regs,
    move_to_regs,
)
from xdsl.dialects import riscv, riscv_func, snitch_runtime
from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr
from xdsl.ir import Attribute, MLContext, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def rewrite_to_func_call(op: Operation, name: str, rewriter: PatternRewriter):
    mv_ops, values = move_to_a_regs(op.operands)
    new_result_types = list(a_regs(op.results))
    func_call = riscv_func.CallOp(SymbolRefAttr(name), values, new_result_types)
    move_result_ops, moved_results = move_to_regs(
        func_call.results,
        (cast(riscv.RISCVRegisterType, v.type) for v in op.results),
    )

    rewriter.replace_op(op, [*mv_ops, func_call, *move_result_ops], moved_results)


class LowerGetInfoOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on all SnitchRuntimeGetInfo ops, since they have
    the same function signature.

    Note: Takes the name of the op and replaces "snrt." with "snrt_" to link with
    snrt.h
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SnitchRuntimeGetInfo, rewriter: PatternRewriter, /
    ):
        rewrite_to_func_call(op, "snrt_" + op.name[5:], rewriter)


class LowerBarrierOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on all SnitchRuntimeBarrier ops, since they have
    the same function signature.

    Note: Takes the name of the op and replaces "snrt." with "snrt_" to link with
    snrt.h
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_runtime.NoOperandNoResultBaseOperation,
        rewriter: PatternRewriter,
        /,
    ):
        rewrite_to_func_call(op, "snrt_" + op.name[5:], rewriter)


class LowerDma1DOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on DmaStart1DOp instances and lowers to external
    function calls.
    Works on both DmaStart1DOp and DmaStart1DWideptrOp.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_runtime.DmaStart1DOp | snitch_runtime.DmaStart1DWideptrOp,
        rewriter: PatternRewriter,
    ):
        rewrite_to_func_call(op, "snrt_" + op.name[5:], rewriter)


class LowerDma2DOpToFunc(RewritePattern, ABC):
    """
    Rewrite pattern that matches on DmaStart2DOp instances and lowers to external
    function calls
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_runtime.DmaStart2DOp | snitch_runtime.DmaStart2DWideptrOp,
        rewriter: PatternRewriter,
    ):
        rewrite_to_func_call(op, "snrt_" + op.name[5:], rewriter)


class AddExternalFuncs(RewritePattern, ABC):
    """
    Looks for snrt ops and adds an external func call to it for LLVM to link in
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: ModuleOp, rewriter: PatternRewriter):
        funcs_to_emit: dict[str, tuple[list[Attribute], list[Attribute]]] = dict()
        for op in module.walk():
            if not isinstance(op, riscv_func.CallOp):
                continue
            if "snrt" not in op.callee.string_value():
                continue
            funcs_to_emit[op.callee.string_value()] = (
                [arg.type for arg in op.args],
                [res.type for res in op.results],
            )

        for name, types in funcs_to_emit.items():
            arg, res = types
            rewriter.insert_op_at_end(
                riscv_func.FuncOp(name, Region(), (arg, res)), module.body.block
            )


class LowerSnitchRuntimePass(ModulePass):
    name = "lower-snrt-to-func"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker1 = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGetInfoOpToFunc(),
                    LowerBarrierOpToFunc(),
                    LowerDma1DOpToFunc(),
                    LowerDma2DOpToFunc(),
                ]
            ),
            apply_recursively=True,
        )
        walker2 = PatternRewriteWalker(AddExternalFuncs())
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
