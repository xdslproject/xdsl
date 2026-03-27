from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType, ModuleOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.loop_nest_lowering_utils import (
    indices_for_map,
    rewrite_linalg_structured_to_loops,
)


class LowerLinalgStructuredOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.LinalgStructuredOperation, rewriter: PatternRewriter
    ) -> None:
        if op.res:
            raise NotImplementedError(
                f"lowering for {op.name} with tensor results not yet supported"
            )

        def insert_load(
            value_index: int,
            ind_vars: Sequence[SSAValue],
            rewriter: PatternRewriter,
            insertion_target: InsertPoint,
        ) -> SSAValue:
            value = op.operands[value_index]
            affine_map_attr = op.get_indexing_maps().data[value_index]
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
            affine_map_attr = op.get_indexing_maps().data[value_index]
            indices = indices_for_map(
                rewriter, insertion_target, affine_map_attr.data, ind_vars
            )
            store_op = memref.StoreOp.get(value, destination, indices)
            rewriter.insert_op(store_op, insertion_target)
            return store_op

        insertion_point = InsertPoint.before(op)
        index = IndexType()
        ub_ops = tuple(
            arith.ConstantOp(IntegerAttr(ub, index))
            for ub in op.get_static_loop_ranges()
        )
        rewriter.insert_op(ub_ops, insertion_point)
        bound_values = tuple(op.result for op in ub_ops)

        rewrite_linalg_structured_to_loops(
            rewriter,
            insertion_point,
            bound_values,
            op.get_indexing_maps().data,
            op.get_indexing_maps().data[-len(op.outputs) :],
            op.operands,
            op.outputs,
            op.body.block,
            insert_load,
            insert_store,
        )


class ConvertLinalgToLoopsPass(ModulePass):
    """
    Converts a linalg structured ops to perfectly nested loops.
    """

    name = "convert-linalg-to-loops"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            LowerLinalgStructuredOpPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
