from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, csl
from xdsl.dialects.csl import csl_wrapper
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    InsertPoint,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def _get_csl_mod(op: Operation) -> csl.CslModuleOp:
    """
    Find the parent `csl.module` of the current op
    """

    if isinstance(op, csl.CslModuleOp):
        return op
    assert (parent := op.parent_op()) is not None
    return _get_csl_mod(parent)


def _collect_ops(op: Operation, ops: list[Operation]):
    """
    Detach the op from its current location and store it in the list

    Do this recursively for all operands of each operation.

    NOTE: This op's dependencies are added to the list first to preserve
          the order in which they get added bakc to the module.
    """
    op.detach()
    for operand in op.operands:
        owner = operand.owner
        assert isinstance(owner, Operation)
        _collect_ops(owner, ops)
    ops.append(op)
    return ops


@dataclass(frozen=True)
class LowerImport(RewritePattern):
    """
    Replace the `csl_wrapper.import` with the equivalent `csl.import`.

    Hoist the import and all ops it depends on to the module scope (as is required by CSL)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ImportOp, rewriter: PatternRewriter, /):
        csl_mod = _get_csl_mod(op)
        ops = _collect_ops(op, [])
        struct = csl.ConstStructOp(*((f.data, o) for f, o in zip(op.fields, op.ops)))
        import_ = csl.ImportModuleConstOp(op.module, struct)

        rewriter.insert_op(ops, InsertPoint.at_start(csl_mod.body.block))
        rewriter.replace_matched_op([struct, import_])


@dataclass(frozen=True)
class CslWrapperLowerImportPass(ModulePass):
    """lower `csl_wrapper.import to csl.import`"""

    name = "csl-wrapper-lower-import"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerImport()]),
            apply_recursively=False,
        ).rewrite_module(op)
