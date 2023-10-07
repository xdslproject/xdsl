"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""


from typing import cast

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import AffineMapAttr, Float64Type, ModuleOp
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Block, MLContext, Region
from xdsl.ir.affine.affine_map import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

from ..dialects import toy
from .lower_toy_affine import (
    ConstantOpLowering,
    FuncOpLowering,
    PrintOpLowering,
    ReturnOpLowering,
    convert_tensor_to_memref,
    insert_alloc_and_dealloc,
)


def lower_op_to_linalg(
    op: toy.AddOp | toy.MulOp | toy.TransposeOp,
    input_map: AffineMapAttr,
    output_map: AffineMapAttr,
    rewriter: PatternRewriter,
    body: Block,
):
    tensor_type = cast(toy.TensorTypeF64, op.res.type)

    # insert an allocation and deallocation for the result of this operation.
    memref_type = convert_tensor_to_memref(tensor_type)
    alloc = insert_alloc_and_dealloc(memref_type, op, rewriter)

    # Create a nest of affine loops, with one loop per dimension of the shape.
    # The buildAffineLoopNest function takes a callback that is used to construct the body
    # of the innermost loop given a builder, a location and a range of loop induction
    # variables.

    parent_block = op.parent
    assert parent_block is not None

    operands = op.operands

    operand_len = len(operands)

    rewriter.replace_matched_op(
        linalg.Generic(
            operands,
            (alloc.memref,),
            Region(body),
            [input_map] * operand_len + [output_map],
            [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)]
            * (operand_len + len(op.results)),
        ),
        new_results=(alloc.memref,),
    )


class AddOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.AddOp, rewriter: PatternRewriter):
        assert isa(
            op.lhs.type, toy.TensorTypeF64 | MemRefType[Float64Type]
        ), f"{op.lhs.type}"
        rank = len(op.lhs.type.shape)
        map_attr = AffineMapAttr(AffineMap.identity(rank))

        block = Block(arg_types=(Float64Type(), Float64Type()))

        with ImplicitBuilder(block) as (l, r):
            s = arith.Addf(l, r)
            linalg.Yield(s)

        lower_op_to_linalg(
            op,
            map_attr,
            map_attr,
            rewriter,
            block,
        )


class MulOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.MulOp, rewriter: PatternRewriter):
        assert isa(
            op.lhs.type, toy.TensorTypeF64 | MemRefType[Float64Type]
        ), f"{op.lhs.type}"
        rank = len(op.lhs.type.shape)
        map_attr = AffineMapAttr(AffineMap.identity(rank))

        block = Block(arg_types=(Float64Type(), Float64Type()))

        with ImplicitBuilder(block) as (l, r):
            p = arith.Mulf(l, r)
            linalg.Yield(p)

        lower_op_to_linalg(
            op,
            map_attr,
            map_attr,
            rewriter,
            block,
        )


class TransposeOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.TransposeOp, rewriter: PatternRewriter):
        assert isa(
            op.arg.type, toy.TensorTypeF64 | MemRefType[Float64Type]
        ), f"{op.arg.type}"
        rank = len(op.arg.type.shape)
        map = AffineMap.identity(rank)

        block = Block(arg_types=(op.arg.type.element_type,))

        with ImplicitBuilder(block) as (i,):
            linalg.Yield(i)

        lower_op_to_linalg(
            op,
            AffineMapAttr(map),
            AffineMapAttr(map.transpose),
            rewriter,
            block,
        )


class LowerToLinalgPass(ModulePass):
    """
    A pass for lowering operations in the Toy dialect to Builtin.
    """

    name = "toy-to-linalg"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    ConstantOpLowering(),
                    FuncOpLowering(),
                    MulOpLowering(),
                    PrintOpLowering(),
                    ReturnOpLowering(),
                    TransposeOpLowering(),
                ]
            )
        ).rewrite_module(op)
