from collections.abc import Sequence
from itertools import compress

from xdsl.dialects import affine, arith, linalg, memref, scf
from xdsl.dialects.builtin import (
    AffineMapAttr,
    IndexType,
    IntegerAttr,
    ModuleOp,
)
from xdsl.ir import Block, BlockArgument, MLContext, Operation, Region, SSAValue
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
    rewriter: PatternRewriter,
    target_op: Operation,
    affine_map: AffineMap,
    input_index_vals: Sequence[SSAValue],
) -> Sequence[SSAValue]:
    if affine_map.num_symbols:
        raise NotImplementedError("Cannot create indices for affine map with symbols")
    output_indices: list[SSAValue] = []
    for expr in affine_map.results:
        if isinstance(expr, AffineDimExpr):
            output_indices.append(input_index_vals[expr.position])
        else:
            used_dims = expr.used_dims()
            new_index_vals = input_index_vals
            new_affine_map = AffineMap(
                affine_map.num_dims, affine_map.num_symbols, (expr,)
            )
            if len(used_dims) != affine_map.num_dims:
                # Remove unused dims
                selectors = tuple(
                    dim in used_dims for dim in range(affine_map.num_dims)
                )
                new_index_vals = tuple(compress(new_index_vals, selectors))
                new_affine_map = new_affine_map.compress_dims(selectors)

            rewriter.insert_op_before(
                apply_op := affine.ApplyOp(
                    new_index_vals,
                    AffineMapAttr(new_affine_map),
                ),
                target_op,
            )

            output_indices.append(apply_op.result)

    return output_indices


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter) -> None:
        if op.res:
            raise NotImplementedError(
                "lowering for linalg.generic with results not yet supported"
            )

        # Create loop nest

        ubs = op.get_static_loop_ranges()

        bound_constant_ops = tuple(
            arith.Constant(IntegerAttr.from_index_int_value(ub)) for ub in ubs
        )
        rewriter.insert_op_before_matched_op(bound_constant_ops)
        bound_constant_values = tuple(op.result for op in bound_constant_ops)

        zero_op = arith.Constant(IntegerAttr.from_index_int_value(0))
        one_op = arith.Constant(IntegerAttr.from_index_int_value(1))
        zero_val = zero_op.result
        one_val = one_op.result
        if bound_constant_values:
            rewriter.insert_op_before_matched_op((zero_op, one_op))

        index = IndexType()

        # Insert loop nest

        loop_args: list[BlockArgument] = []
        insertion_target: Operation = op

        for ub in bound_constant_values:
            loop = scf.For(
                zero_val,
                ub,
                one_val,
                (),
                Region(Block((yield_op := scf.Yield(),), arg_types=(index,))),
            )
            loop_args.append(loop.body.block.args[0])
            rewriter.insert_op_before(loop, insertion_target)
            insertion_target = yield_op

        # Add load ops

        for affine_map_attr, operand, arg in zip(
            op.indexing_maps.data, op.operands, op.body.block.args, strict=True
        ):
            if not arg.uses:
                continue
            affine_map = affine_map_attr.data
            indices = indices_for_map(rewriter, insertion_target, affine_map, loop_args)
            load_op = memref.Load.get(operand, indices)
            rewriter.insert_op_before(load_op, insertion_target)
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
            indices = indices_for_map(rewriter, insertion_target, affine_map, loop_args)
            store_op = memref.Store.get(yield_value, ref, indices)
            rewriter.insert_op_before(store_op, linalg_yield_op)

        rewriter.erase_op(linalg_yield_op)

        # Inline generic body

        rewriter.inline_block_before(op.body.block, insertion_target)

        # Erase generic

        rewriter.erase_matched_op()


class ConvertLinalgToLoopsPass(ModulePass):
    name = "convert-linalg-to-loops"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
