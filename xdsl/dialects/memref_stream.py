"""
This work-in-progress dialect represents a target-independent representation of streams of
registers over time. It's missing an operation and an attribute:

 - memref_stream.stride_pattern:
    attribute specifying the order in which elements are accessed
 - memref_stream.streaming_region:
    taking streams and stride patterns, and providing the stream values to be read from or written to

These, along with more detailed documentation, will be added over the next week.
"""

from __future__ import annotations

from xdsl.dialects import stream
from xdsl.ir import Dialect
from xdsl.irdl import irdl_op_definition


@irdl_op_definition
class ReadOp(stream.ReadOperation):
    name = "memref_stream.read"


@irdl_op_definition
class WriteOp(stream.WriteOperation):
    name = "memref_stream.write"


MemrefStream = Dialect(
    "memref_stream",
    [
        ReadOp,
        WriteOp,
    ],
    [],
)
