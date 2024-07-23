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
from xdsl.ir import Attribute
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

StreamingVectorType: TypeAlias = VectorType[Float64Type | Float32Type | Float16Type]
StreamingLegalType: TypeAlias = StreamingVectorType | Float64Type


def _is_legal_vector(attr: Attribute) -> bool:
    if isa(attr, StreamingVectorType):
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
    return _is_legal_vector(attr) or isa(attr, StreamingLegalType)


def _legalize_attr(attr: Attribute) -> StreamingLegalType:
    if isa(attr, StreamingVectorType):
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


@dataclass(frozen=True)
class MemrefStreamGenericLegalize(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # FIXME add reduction support
        if op.iterator_types.data[-1].data != IteratorType.PARALLEL:
            return
        # Collect block arguments that need to be legalized
        legalizations: dict[int, Attribute] = {
            i: _legalize_attr(arg.type)
            for i, arg in enumerate(op.body.block.args)
            if not _is_legal_attr(arg.type)
        }
        if len(legalizations) == 0:
            return
        # Check that vectorized bounds are compatible with all no. of lanes
        # involved in legalizations
        innermost_bound = op.bounds.data[-1].value.data
        vector_lengths = set()
        for i, v in legalizations.items():
            n_lanes = v.get_shape()[0]
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
        # Legalize payload
        new_body = op.body.clone()
        for i, arg in enumerate(new_body.block.args):
            if i not in legalizations:
                continue
            rewriter.modify_block_argument_type(arg, legalizations[i])
            # Fixup results of ops that use this as operand
            for use in tuple(arg.uses):
                use_op = use.operation
                new_op = use_op.create(
                    operands=use_op.operands,
                    result_types=[_legalize_attr(res.type) for res in use_op.results],
                    attributes=use_op.attributes,
                )
                rewriter.replace_op(use_op, new_op)
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
