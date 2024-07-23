from dataclasses import dataclass
from typing import TypeAlias

from xdsl.context import MLContext
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    IntegerAttr,
    ModuleOp,
    VectorType,
    f16,
    f32,
)
from xdsl.dialects.linalg import IteratorType
from xdsl.ir import Attribute, Block, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa

StreamingVectorLegalizationType: TypeAlias = VectorType[
    Float64Type | Float32Type | Float16Type
]
StreamingAlreadyLegalType: TypeAlias = Float64Type


def _is_legal_vector(attr: Attribute) -> bool:
    if isa(attr, StreamingVectorLegalizationType):
        match attr.element_type, attr.element_count():
            case Float64Type(), 1:
                return True
            case Float32Type(), 2:
                return True
            case Float16Type(), 4:
                return True
            case _:
                return False
    return False


def _is_legal_attr(attr: Attribute) -> bool:
    return _is_legal_vector(attr) or isinstance(attr, StreamingAlreadyLegalType)


def _legalize_attr(
    attr: Attribute,
) -> StreamingVectorLegalizationType | StreamingAlreadyLegalType:
    if isa(attr, StreamingVectorLegalizationType):
        # Either already a legal vector or impossible to legalize
        if not _is_legal_vector(attr):
            raise DiagnosticException(f"Cannot legalize {attr} for streaming")
        return attr
    elif isinstance(attr, Float64Type):
        return attr
    elif isinstance(attr, Float32Type):
        return VectorType(f32, (2,))
    elif isinstance(attr, Float16Type):
        return VectorType(f16, (4,))
    else:
        raise DiagnosticException(f"Cannot legalize {attr} for streaming")


def _legalize_block(block: Block, rewriter: PatternRewriter) -> None:
    # Start from all uses of all block arguments
    to_be_legalized: set[Operation] = {
        use.operation for arg in block.args for use in arg.uses
    }
    # Linearly scan the block and update the set of ops that we need to look into
    # by following uses
    for op in block.ops:
        if op not in to_be_legalized:
            continue
        to_be_legalized.remove(op)
        illegal_results: list[int] = [
            result.index for result in op.results if not _is_legal_attr(result.type)
        ]
        if not illegal_results:
            continue
        new_op = op.create(
            operands=op.operands,
            result_types=[_legalize_attr(res.type) for res in op.results],
            attributes=op.attributes,
        )
        rewriter.replace_op(op, new_op)
        for idx in illegal_results:
            to_be_legalized.update(use.operation for use in new_op.results[idx].uses)


@dataclass(frozen=True)
class MemrefStreamGenericLegalize(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # Collect block arguments that need to be legalized
        legalizations: dict[int, StreamingVectorLegalizationType] = {}
        for i, arg in enumerate(op.body.block.args):
            legal = _legalize_attr(arg.type)
            if not isinstance(legal, StreamingAlreadyLegalType):
                legalizations[i] = legal
        if not legalizations:
            return
        if op.iterator_types.data[-1].data != IteratorType.PARALLEL:
            raise DiagnosticException(
                "iterators other than 'parallel' are not supported yet"
            )
        # Check that vectorized bounds are compatible with all no. of lanes
        # involved in legalizations
        innermost_bound = op.bounds.data[-1].value.data
        vector_lengths: set[int] = set()
        for i, v in legalizations.items():
            n_lanes: int = v.get_shape()[0]
            if innermost_bound % n_lanes != 0:
                raise ValueError(
                    f"no. of vector lanes ({n_lanes}) introduced to legalize argument #{i} "
                    f"is not a divisor for the innermost dimension's bound ({innermost_bound})"
                )
            vector_lengths.add(n_lanes)
        if len(vector_lengths) != 1:
            # FIXME we should deal with heterogeneous generic ops
            raise NotImplementedError(
                "cannot legalize heterogeneous block arguments yet"
            )
        vlen = next(iter(vector_lengths))
        # Legalize iteration bounds
        new_bounds = list(op.bounds)
        new_bounds.pop()
        new_bounds.append(IntegerAttr.from_index_int_value(innermost_bound // vlen))
        # Legalize block arguments
        new_body = op.body.clone()
        for i, arg in enumerate(new_body.block.args):
            if i not in legalizations:
                continue
            rewriter.modify_block_argument_type(arg, legalizations[i])
        # Legalize payload
        _legalize_block(new_body.block, rewriter)

        rewriter.replace_matched_op(
            memref_stream.GenericOp(
                op.inputs,
                op.outputs,
                op.inits,
                new_body,
                op.indexing_maps,
                op.iterator_types,
                ArrayAttr(new_bounds),
                op.init_indices,
            )
        )


@dataclass(frozen=True)
class MemrefStreamLegalizePass(ModulePass):
    """
    Legalize memref_stream.generic payload and bounds for streaming.
    """

    name = "memref-stream-legalize"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([MemrefStreamGenericLegalize()]),
            apply_recursively=False,
        ).rewrite_module(op)
