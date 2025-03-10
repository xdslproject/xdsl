from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import linalg, memref
from xdsl.dialects.builtin import MemRefType, ModuleOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.loop_nest_lowering_utils import (
    indices_for_map,
    rewrite_generic_to_loops,
)


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if op.res:
            raise NotImplementedError(
                "lowering for linalg.generic with results not yet supported"
            )

        def insert_load(
            value_index: int,
            ind_vars: Sequence[SSAValue],
            rewriter: PatternRewriter,
            insertion_target: InsertPoint,
        ) -> SSAValue:
            value = op.operands[value_index]
            affine_map_attr = op.indexing_maps.data[value_index]
            if isinstance(value.type, MemRefType):
                indices = indices_for_map(
                    rewriter, insertion_target, affine_map_attr.data, ind_vars
                )
                load_op = memref.LoadOp.get(value, indices)
                rewriter.insert_op(load_op, insertion_target)
                return load_op.res
            else:
                return value

        ins_count = len(op.inputs)

        def insert_store(
            output_index: int,
            value: SSAValue,
            ind_vars: Sequence[SSAValue],
            rewriter: PatternRewriter,
            insertion_target: InsertPoint,
        ):
            value_index = ins_count + output_index
            destination = op.operands[value_index]
            affine_map_attr = op.indexing_maps.data[value_index]
            indices = indices_for_map(
                rewriter, insertion_target, affine_map_attr.data, ind_vars
            )
            store_op = memref.StoreOp.get(value, destination, indices)
            rewriter.insert_op(store_op, insertion_target)
            return store_op

        rewrite_generic_to_loops(
            rewriter,
            InsertPoint.before(op),
            op.get_static_loop_ranges(),
            op.indexing_maps.data,
            op.indexing_maps.data[-len(op.outputs) :],
            op.operands,
            op.outputs,
            op.body.block,
            insert_load,
            insert_store,
        )


class ConvertLinalgToLoopsPass(ModulePass):
    """
    Converts a linalg generic to perfectly nested loops.
    """

    name = "convert-linalg-to-loops"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
