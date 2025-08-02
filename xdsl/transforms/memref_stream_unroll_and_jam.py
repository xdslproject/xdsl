from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.memref_stream_interleave import PipelineGenericPattern
from xdsl.utils.exceptions import DiagnosticException


@dataclass(frozen=True)
class MemrefStreamUnrollAndJamPass(ModulePass):
    """
    TODO
    """

    name = "memref-stream-unroll-and-jam"

    op_index: int
    iterator_index: int
    unroll_factor: int

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        msg_ops = (
            child for child in op.walk() if isinstance(child, memref_stream.GenericOp)
        )

        msg_op = None

        for index, child in enumerate(msg_ops):
            if index == self.op_index:
                msg_op = child
                break

        if msg_op is None:
            raise DiagnosticException("Index out of bounds")

        PipelineGenericPattern(
            4, self.iterator_index, self.unroll_factor
        ).match_and_rewrite(msg_op, PatternRewriter(msg_op))
