from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, memref, ptr, vector
from xdsl.dialects.builtin import (
    AffineMapAttr,
    FixedBitwidthType,
    IndexType,
    IntegerAttr,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


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

        map_operands = list(op.indices)
        ops: list[Operation] = []

        if isinstance(memory_ty.layout, StridedLayoutAttr) and isinstance(
            memory_ty.layout.offset, NoneAttr
        ):
            ops.append(zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())))
            zero_op.result.name_hint = "c0"
            map_operands.append(zero_op.result)

        ops.append(
            apply_op := affine.ApplyOp(
                map_operands=map_operands,
                affine_map=AffineMapAttr(layout_map),
            )
        )

        # Compute the linearized offset
        ops.append(cast_op := ptr.ToPtrOp(memory))
        ops.append(add_op := ptr.PtrAddOp(cast_op.res, apply_op.result))

        # Load a vector from the pointer
        ops.append(ptr.LoadOp(add_op.result, vector_ty))

        rewriter.replace_matched_op(ops)


@dataclass
class VectorStoreToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.StoreOp, rewriter: PatternRewriter):
        # Input vector description
        assert isa(vector_ty := op.vector.type, vector.VectorType)
        element_type = vector_ty.get_element_type()
        assert isinstance(element_type, FixedBitwidthType)

        # Input memref description
        memory = op.base
        memory_ty = memory.type
        assert isinstance(memory_ty, memref.MemRefType)

        # Build an affine.apply to compute the linearized offset
        layout_map = memory_ty.get_affine_map_in_bytes()

        map_operands = list(op.indices)
        ops: list[Operation] = []

        if isinstance(memory_ty.layout, StridedLayoutAttr) and isinstance(
            memory_ty.layout.offset, NoneAttr
        ):
            ops.append(zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())))
            zero_op.result.name_hint = "c0"
            map_operands.append(zero_op.result)

        ops.append(
            apply_op := affine.ApplyOp(
                map_operands=map_operands,
                affine_map=AffineMapAttr(layout_map),
            )
        )

        # Compute the linearized offset
        ops.append(cast_op := ptr.ToPtrOp(memory))
        ops.append(add_op := ptr.PtrAddOp(cast_op.res, apply_op.result))

        # Store a vector into the pointer
        ops.append(ptr.StoreOp(add_op.result, op.vector))

        rewriter.replace_matched_op(ops)


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
