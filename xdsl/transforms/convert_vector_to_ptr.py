from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, memref, ptr, vector
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    UnrealizedConversionCastOp,
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
        vector_int_shape = [i.data for i in vector_ty.shape.data]
        element_type = vector_ty.get_element_type()
        assert isinstance(element_type, FixedBitwidthType)

        # Input memref description
        memory = op.base
        memory_ty = memory.type
        assert isinstance(memory_ty, memref.MemRefType)

        # Build a subview of the original memref
        strides = memory_ty.get_strides()
        static_strides: list[int] = []
        if strides:
            static_strides += [s for s in strides if s is not None]
        subview_type = memref.MemRefType(
            element_type=element_type, shape=vector_int_shape
        )
        subview_op = memref.SubviewOp.get(
            source=memory,
            result_type=subview_type,
            offsets=op.indices,
            strides=static_strides,
            sizes=vector_int_shape,
        )

        # Build a pointer from the subview
        cast_op = UnrealizedConversionCastOp.get((subview_op.result,), (ptr.PtrType(),))
        # Load a vector from the pointer
        load_op = ptr.LoadOp(operands=cast_op.outputs, result_types=[vector_ty])

        rewriter.replace_matched_op([subview_op, cast_op, load_op])


@dataclass(frozen=True)
class ConvertVectorToPtrPass(ModulePass):
    name = "convert-vector-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([VectorLoadToPtr()]),
            apply_recursively=False,
        ).rewrite_module(op)
