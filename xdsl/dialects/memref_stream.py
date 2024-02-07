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
