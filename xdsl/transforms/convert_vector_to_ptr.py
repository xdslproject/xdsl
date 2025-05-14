from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import affine, builtin, memref, ptr, vector
from xdsl.dialects.builtin import (
    AffineMapAttr,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def linearize_indices(
    memory: SSAValue,
    indices: Sequence[SSAValue[Attribute]],
) -> affine.ApplyOp:
    memory_ty = memory.type
    assert isinstance(memory_ty, memref.MemRefType)
    layout_map = memory_ty.get_affine_map_in_bytes()
    apply_op = affine.ApplyOp(
        map_operands=indices,
        affine_map=AffineMapAttr(layout_map),
    )
    return apply_op


@dataclass
class VectorStoreToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.StoreOp, rewriter: PatternRewriter):
        # Compute the linearized offset
        apply_op = linearize_indices(memory=op.base, indices=op.indices)
        cast_op = ptr.ToPtrOp(op.base)
        add_op = ptr.PtrAddOp(cast_op.res, apply_op.result)

        # Store the vector at the memory location
        store_op = ptr.StoreOp(addr=add_op.result, value=op.vector)

        rewriter.replace_matched_op([apply_op, cast_op, add_op, store_op])


@dataclass
class VectorLoadToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.LoadOp, rewriter: PatternRewriter):
        # Compute the linearized offset
        apply_op = linearize_indices(memory=op.base, indices=op.indices)
        cast_op = ptr.ToPtrOp(op.base)
        add_op = ptr.PtrAddOp(cast_op.res, apply_op.result)

        # Load a vector from the pointer
        load_op = ptr.LoadOp(add_op.result, op.result.type)

        rewriter.replace_matched_op([apply_op, cast_op, add_op, load_op])


@dataclass(frozen=True)
class ConvertVectorToPtrPass(ModulePass):
    name = "convert-vector-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    VectorLoadToPtr(),
                    VectorStoreToPtr(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
