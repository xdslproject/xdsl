from collections.abc import Sequence

from xdsl.builder import Builder
from xdsl.dialects import arith, linalg, memref, scf
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    ModuleOp,
)
from xdsl.ir import Block, BlockArgument, MLContext, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def indices_for_map(
    affine_map: AffineMap, input_index_vals: Sequence[SSAValue]
) -> Sequence[SSAValue]:
    output_indices: list[int] = []
    for expr in affine_map.results:
        if not isinstance(expr, AffineDimExpr):
            raise NotImplementedError("Cannot handle non-dim affine maps")
        output_indices.append(expr.position)

    return tuple(input_index_vals[index] for index in output_indices)


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter) -> None:
        if op.res:
            raise NotImplementedError(
                "lowering for linalg.generic with results not yet supported"
            )

        # Create loop nest

        ubs = op.get_static_loop_ranges()

        if not ubs:
            assert False, "TODO"

        bound_constant_ops = tuple(
            arith.Constant(IntegerAttr.from_index_int_value(ub)) for ub in ubs
        )
        rewriter.insert_op_before_matched_op(bound_constant_ops)
        bound_constant_values = tuple(op.result for op in bound_constant_ops)

        zero_op = arith.Constant(IntegerAttr.from_index_int_value(0))
        one_op = arith.Constant(IntegerAttr.from_index_int_value(1))
        rewriter.insert_op_before_matched_op((zero_op, one_op))
        zero_val = zero_op.result
        one_val = one_op.result

        index = IndexType()

        loop_args: list[BlockArgument] = []

        innermost_loop = scf.For(
            zero_val,
            bound_constant_values[-1],
            one_val,
            (),
            Region(Block((scf.Yield(),), arg_types=(index,))),
        )
        innermost_body = innermost_loop.body.block

        outermost_loop = innermost_loop
        loop_args = [innermost_body.args[0]]

        for ub in reversed(bound_constant_values[:-1]):
            outermost_loop = scf.For(
                zero_val,
                ub,
                one_val,
                (),
                Region(Block((outermost_loop, scf.Yield()), arg_types=(index,))),
            )
            loop_args.append(outermost_loop.body.block.args[0])

        # last loop is innermost
        loop_args.reverse()

        # Add load ops

        b = Builder.at_start(innermost_body)

        for affine_map_attr, operand, arg in zip(
            op.indexing_maps.data, op.operands, op.body.block.args, strict=True
        ):
            if not arg.uses:
                continue
            affine_map = affine_map_attr.data
            indices = indices_for_map(affine_map, loop_args)
            load_op = memref.Load.get(operand, indices)
            b.insert(load_op)
            arg.replace_by(load_op.res)

        # Add store ops

        linalg_yield_op = op.body.block.last_op
        assert isinstance(linalg_yield_op, linalg.YieldOp)

        output_indexing_maps = op.indexing_maps.data[-len(op.outputs) :]
        output_operands = op.operands[-len(op.outputs) :]
        for affine_map_attr, yield_value, ref in zip(
            output_indexing_maps, linalg_yield_op.operands, output_operands, strict=True
        ):
            affine_map = affine_map_attr.data
            indices = indices_for_map(affine_map, loop_args)
            store_op = memref.Store.get(yield_value, ref, indices)
            rewriter.insert_op_before(store_op, linalg_yield_op)

        rewriter.erase_op(linalg_yield_op)

        # Insert loop nest

        rewriter.insert_op_before_matched_op(outermost_loop)

        # Inline generic body

        scf_yield_op = innermost_body.last_op
        assert scf_yield_op is not None
        rewriter.inline_block_before(op.body.block, scf_yield_op)

        # Erase generic

        rewriter.erase_matched_op()


class ConvertLinalgToLoopsPass(ModulePass):
    name = "convert-linalg-to-loops"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
