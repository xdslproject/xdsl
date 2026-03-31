from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.csl import csl, csl_wrapper
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


@dataclass(frozen=True)
class HoistBuffers(RewritePattern):
    """
    Hoists buffers to `csl_wrapper.program_module`-level.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter, /):
        # always attempt to set name hints
        self._set_name_hint(op)

        wrapper = op.parent_op()
        while wrapper and not isinstance(wrapper, csl_wrapper.ModuleOp):
            wrapper = wrapper.parent_op()

        # no action required if this op exists on module-level
        if not wrapper or wrapper == op.parent_op():
            return

        assert len(op.dynamic_sizes) == 0, "not implemented"
        assert len(op.symbol_operands) == 0, "not implemented"

        rewriter.insert_op(
            alloc := op.clone(), InsertPoint.at_start(wrapper.program_module.block)
        )
        rewriter.replace_op(op, [], new_results=[alloc.memref])

    @staticmethod
    def _set_name_hint(op: memref.AllocOp):
        """
        Attempts to find a chain of:
          %0 = memref.alloc
          %1 = csl.addressof(%0)
          csl.export(%1) <{var_name = "buf"}>

        and sets name hints for alloc and addressof to "buf" and "buf_ptr", respectively
        """
        for ptr_use in op.memref.uses:
            if not isinstance(ptr_op := ptr_use.operation, csl.AddressOfOp):
                continue

            for exp_use in ptr_op.res.uses:
                if not isinstance(exp_op := exp_use.operation, csl.SymbolExportOp):
                    continue
                op.memref.name_hint = exp_op.get_name()
                ptr_op.res.name_hint = f"{exp_op.get_name()}_ptr"
                return


@dataclass(frozen=True)
class CslWrapperHoistBuffers(ModulePass):
    """
    Hoists buffers to the `csl_wrapper.program_module`-level.
    """

    name = "csl-wrapper-hoist-buffers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(HoistBuffers())
        module_pass.rewrite_module(op)
