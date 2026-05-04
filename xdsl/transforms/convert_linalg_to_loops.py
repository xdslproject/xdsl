from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IndexType,
    MemRefType,
    ModuleOp,
)
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
from xdsl.utils.exceptions import PassFailedException
from xdsl.utils.hints import isa


def materialize_loop_bound(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    operand: SSAValue[MemRefType],
    dim_index: int,
    dim_size: int,
) -> SSAValue:
    """
    Create the value used as one loop upper bound.

    If the memref dimension is dynamic, use `memref.dim`.
    If it is static, use an index constant.
    """

    if dim_size == DYNAMIC_INDEX:
        dim_index_op = arith.ConstantOp.from_int_and_width(dim_index, IndexType())
        rewriter.insert_op(dim_index_op, insertion_point)

        dim_op = memref.DimOp.from_source_and_index(operand, dim_index_op.result)
        rewriter.insert_op(dim_op, insertion_point)
        return dim_op.result

    else:
        const_op = arith.ConstantOp.from_int_and_width(dim_size, IndexType())
        rewriter.insert_op(const_op, insertion_point)
        return const_op.result


def create_loop_bounds(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    op: linalg.LinalgStructuredOperation,
) -> Sequence[SSAValue]:
    """
    Build loop upper bounds for a linalg structured operation.

    This lowering only supports buffer semantics, so bound sources must come
    from memref operands.
    """
    bounds: list[SSAValue] = []

    for operand, dim_index, dim_size in op.get_loop_bound_sources():
        if not isa(operand, SSAValue[MemRefType]):
            raise PassFailedException(
                "convert-linalg-to-loops requires buffer semantics; "
                "tensor operands must be bufferized to memrefs before lowering"
            )

        bounds.append(
            materialize_loop_bound(
                rewriter,
                insertion_point,
                operand,
                dim_index,
                dim_size,
            )
        )

    return bounds


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
        rewrite_linalg_structured_to_loops(
            rewriter,
            insertion_point,
            create_loop_bounds(rewriter, insertion_point, op),
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
