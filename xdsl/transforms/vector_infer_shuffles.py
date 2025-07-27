from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, vector
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


@dataclass
class FuseExtractBroadcastPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.BroadcastOp, rewriter: PatternRewriter, /):
        scalar = op.source
        if isa(scalar.type, builtin.VectorType):
            # Only do 1-D vector -> scalar -> 1-D vector
            return

        if not isinstance(extract_op := op.source.owner, vector.ExtractOp):
            return

        if extract_op.dynamic_position:
            # Cannot represent dynamic positions in shuffle
            return

        source = extract_op.vector
        source_type = source.type
        assert isa(source_type, builtin.VectorType)

        if source_type.get_num_scalable_dims():
            # Scalable dims not supported in shuffle op
            return

        static_dims = extract_op.static_position.get_values()
        if len(static_dims) != 1:
            # Only support 1-D source vectors
            return

        destination = op.vector
        destination_type = destination.type
        destination_shape = destination_type.get_shape()
        if len(destination_shape) != 1:
            # Only support 1-D destination vectors
            return

        index = static_dims[0]
        shuffle_indices = (index,) * destination_shape[0]

        rewriter.replace_matched_op(
            vector.ShuffleOp(
                source,
                source,
                builtin.DenseArrayBase.from_list(builtin.i64, shuffle_indices),
                result_type=op.vector.type,
            )
        )


class VectorInferShufflesPass(ModulePass):
    """
    Finds occurences of a vector extract followed by a vector broadcast and converts the
    result to a vector suffle.
    """

    name = "vector-infer-shuffles"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            FuseExtractBroadcastPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
