from xdsl.dialects import affine, arith, memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..dialects import toy_accelerator


class LowerAffineForOp(RewritePattern):
    """
    Matches on nested loops that implement elementwise addition or transpose, and rewrites
    them as custom toy accelerator instructions.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.For, rewriter: PatternRewriter):
        # Check this is a doubly nested affine for loop, where both loops don't yield
        # values.
        if (
            len(op.body.blocks) != 1
            or len((outer_block := op.body.blocks[0]).ops) != 2
            or not isinstance(inner_for := outer_block.first_op, affine.For)
            or not isinstance(outer_block.last_op, affine.Yield)
            or outer_block.last_op.operands
            or len(inner_for.body.blocks) != 1
            or not (inner_block := inner_for.body.blocks[0])
            or not isinstance(inner_block.last_op, affine.Yield)
            or inner_block.last_op.operands
        ):
            return

        # Check that the indices step from 0 to upper bound by 1

        all_bounds = (
            op.lower_bound.data,
            op.upper_bound.data,
            inner_for.lower_bound.data,
            inner_for.upper_bound.data,
        )
        if any(bound.num_dims or bound.num_symbols for bound in all_bounds):
            # Loop is not made of constant values
            return

        ostep = op.step.value.data
        istep = inner_for.step.value.data

        if ostep != 1 or istep != 1:
            # Loop does not step by 1
            return

        olb, oub, ilb, iub = tuple(bound.eval([], [])[0] for bound in all_bounds)

        if olb != 0 or ilb != 0:
            # Loop doesn't start from 0
            return

        # We now need to check whether it's a transpose or an elementwise
        # operation.

        if len(inner_block.ops) == 3:
            (load, store, _) = inner_block.ops
            if not (isinstance(load, affine.Load) and isinstance(store, affine.Store)):
                return

            if load.indices != store.indices[::-1]:
                return

            # This is a transpose op

            memref_typ = load.memref.type
            assert isinstance(memref_typ, memref.MemRefType)
            rows, cols = memref_typ.shape

            if rows.value.data != iub or cols.value.data != oub:
                # The indices aren't fully enumerated
                return

            rewriter.replace_matched_op(
                toy_accelerator.Transpose(store.memref, load.memref, rows, cols)
            )
        elif len(inner_block.ops) == 5:
            lhs_load, rhs_load, binop, store, _ = inner_block.ops
            if not (
                isinstance(lhs_load, affine.Load)
                and isinstance(rhs_load, affine.Load)
                and isinstance(store, affine.Store)
                and isinstance(binop, (arith.Addf, arith.Mulf))
            ):
                return

            # size = oub * iub

            instr_cls = (
                toy_accelerator.Add
                if isinstance(binop, arith.Addf)
                else toy_accelerator.Mul
            )

            rewriter.replace_matched_op(
                [instr_cls(store.memref, lhs_load.memref, rhs_load.memref)]
            )


class LowerToToyAccelerator(ModulePass):
    name = "lower-to-toy-accelerator"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        ctx.register_dialect(toy_accelerator.ToyAccelerator)
        PatternRewriteWalker(LowerAffineForOp()).rewrite_module(op)
