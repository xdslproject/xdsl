from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import affine, builtin, memref, ptr, vector
from xdsl.dialects.builtin import (
    AffineMapAttr,
    FixedBitwidthType,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class VectorLoadToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.LoadOp, rewriter: PatternRewriter):
        # Output vector description
        vector_ty = op.result.type
        element_type = vector_ty.get_element_type()
        assert isinstance(element_type, FixedBitwidthType)

        # Input memref description
        memory = op.base
        memory_ty = memory.type
        assert isinstance(memory_ty, memref.MemRefType)

        # Build an affine.apply to compute the linearized offset
        layout_map = memory_ty.get_affine_map_in_bytes()
        apply_op = affine.ApplyOp(
            map_operands=op.indices,
            affine_map=AffineMapAttr(layout_map),
        )

        # Compute the linearized offset
        cast_op = ptr.ToPtrOp(memory)
        add_op = ptr.PtrAddOp(cast_op.res, apply_op.result)

        # Load a vector from the pointer
        load_op = ptr.LoadOp(add_op.result, vector_ty)

        rewriter.replace_matched_op([apply_op, cast_op, add_op, load_op])


@dataclass(frozen=True)
class ConvertVectorToPtrPass(ModulePass):
    name = "convert-vector-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    VectorLoadToPtr(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
