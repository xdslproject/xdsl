from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg, memref, vector
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa

_index_type = builtin.IndexType()


@dataclass
class VectorizeMatmulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MatmulOp, rewriter: PatternRewriter, /):
        # C += A * B
        # C: M x N, A: M x K, B: K x N

        a, b = op.inputs
        c = op.outputs[0]

        a_type = a.type
        b_type = b.type
        c_type = c.type

        # Only handle matmul on memrefs for now
        if (
            not isa(a_type, builtin.MemRefType)
            or not isa(b_type, builtin.MemRefType)
            or not isa(c_type, builtin.MemRefType)
        ):
            raise DiagnosticException(
                "Vectorizing matmul on tensors not yet implemented."
            )

        M, K = a_type.get_shape()
        _K, N = b_type.get_shape()
        _M, _N = c_type.get_shape()

        assert M == _M
        assert N == _N
        assert K == _K
        assert M != -1
        assert N != -1
        assert K != -1

        vector_type = builtin.VectorType(a_type.element_type, (N,))

        # All operations created inside this block are inserted before the matched op
        with ImplicitBuilder(rewriter):
            # Insert all the integer constants we'll need to index into the matrices
            constants = tuple(
                arith.ConstantOp(builtin.IntegerAttr(i, _index_type)).result
                for i in range(max(M, N, K))
            )
            # Zero for convenience
            c0 = constants[0]

            # Load the rows of C as vectors
            c_rows = [
                vector.LoadOp(c, (constants[m], c0), vector_type).result
                for m in range(M)
            ]

            # Load the rows of B as vectors
            b_rows = tuple(
                vector.LoadOp(b, (constants[k], c0), vector_type).result
                for k in range(K)
            )

            for m in range(M):
                # Load the mth column of A as scalars
                a_col = tuple(
                    memref.LoadOp.get(a, (constants[m], constants[k])).res
                    for k in range(K)
                )
                # Broadcast the mth column of A to vectors
                a_col_vectors = tuple(
                    vector.BroadcastOp(a_col[k], vector_type) for k in range(K)
                )

                for k in range(K):
                    # Accumulate the dot product of rows of B with A's column
                    # The list c_rows is updated in place for convenience, but we're
                    # really creating a new SSA value on each iteration
                    c_rows[m] = vector.FMAOp(a_col_vectors[k], b_rows[k], c_rows[m]).res

            for m in range(M):
                vector.StoreOp(c_rows[m], c, (constants[m], c0))

        rewriter.erase_op(op)


@dataclass(frozen=True)
class TestVectorizeMatmulPass(ModulePass):
    """
    A test pass vectorizing linalg.matmul with a specific vectorization strategy.
    """

    name = "test-vectorize-matmul"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            VectorizeMatmulOp(), apply_recursively=False
        ).rewrite_module(op)
