from __future__ import annotations

from typing import Annotated, Any

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyAttr,
    ArrayAttr,
    Float32Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    i8,
    i32,
    i64,
)
from xdsl.ir import Data, Dialect, OpResult
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer

CASCADE_SIZE = 384


@irdl_attr_definition
class WireBundleAttr(Data[str]):
    name = "wire_bundle"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')


@irdl_op_definition
class AMSelOp(IRDLOperation):
    name = "amsel"
    arbiterID: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    msel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    result: OpResult = result_def(IndexType())

    def __init__(self, arbiterID: IntegerAttr[i32], msel: IntegerAttr[i32]):
        super().__init__(
            attributes={"arbiterID": arbiterID, "msel": msel},
            result_types=[IndexType()],
        )


@irdl_op_definition
class BufferOp(IRDLOperation):
    name = "buffer"
    tile: Operand = operand_def(IndexType())
    shape: ArrayAttr[i32] = attr_def(ArrayAttr[i32])
    element_type: AnyAttr = attr_def(AnyAttr())
    buffer: OpResult = result_def(memref.MemRefType)

    def __init__(self, tile: IndexType(), element_type: AnyAttr, shape: ArrayAttr[i32]):
        buffer_type = memref.MemRefType.from_element_type_and_shape(element_type, shape)
        super().__init__(operands=[tile], result_types=[buffer_type])


@irdl_op_definition
class TileOp(IRDLOperation):
    name = "tile"
    col: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    row: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    result: OpResult = result_def(IndexType())

    def __init__(self, col: IntegerAttr[i32], row: IntegerAttr[i32]):
        super().__init__(
            attributes={"col": col, "row": row}, result_types=[IndexType()]
        )


@irdl_op_definition
class ConnectOp(IRDLOperation):
    name = "connect"
    sourceBundle: WireBundleAttr = attr_def(WireBundleAttr)
    sourceChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    destBundle: WireBundleAttr = attr_def(WireBundleAttr)
    destChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(
        self,
        sourceBundle: WireBundleAttr,
        sourceChannel: IntegerAttr[i32],
        destBundle: WireBundleAttr,
        destChannel: IntegerAttr[i32],
    ):
        super().__init__(
            attributes={
                "sourceBundle": sourceBundle,
                "sourceChannel": sourceChannel,
                "destBundle": destBundle,
                "destChannel": destChannel,
            }
        )


@irdl_op_definition
class CoreOp(IRDLOperation):
    name = "core"
    stackSize: IntegerAttr[i32] = attr_def(IntegerAttr)
    tile: Operand = operand_def(IndexType())

    def __init__(self, stackSize: IntegerAttr[i32], tile: IndexType()):
        super().__init__(
            attributes={"stackSize": stackSize},
            operands=[tile],
            result_types=[IndexType()],
        )


"""
@irdl_op_definition
class DMABDOp(IRDLOperation):
	name = "dmaBd"
	offset : IntegerAttr[i32] = attr_def(IntegerAttr[i32])
	length : IntegerAttr[i32] = attr_def(IntegerAttr[i32])
	AB : IntegerAttr[i32] = attr_def(IntegerAttr[i32])
	dimensions : IntegerAttr[i32] = attr_def(IntegerAttr[i32])
	buffer : Operand = operand_def(BufferOp)
"""


@irdl_op_definition
class DMABDPACKETOp(IRDLOperation):
    name = "dmaBdPacket"
    packet_type: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    packet_id: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(self, packet_type: IntegerAttr[i32], packet_id: IntegerAttr[i32]):
        super().__init__(
            attributes={"packet_type": packet_type, "packet_id": packet_id}
        )


# TODO: add successor basic blocks
@irdl_op_definition
class DMAStartOp(IRDLOperation):
    name = "dmaStart"
    channelDir: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    channelIndex: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(self, channelDir: IntegerAttr[i32], channelIndex: IntegerAttr[i32]):
        super().__init__(
            attributes={"channelDir": channelDir, "channelIndex": channelIndex},
            result_types=[i32],
        )


@irdl_op_definition
class DebugOp(IRDLOperation):
    name = "debug"
    arg: Operand = operand_def(Any)

    def __init__(self, arg: Any):
        super().__init__(operands=[arg])


@irdl_op_definition
class DeviceOp(IRDLOperation):
    name = "device"
    device: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )


@irdl_op_definition
class end(IRDLOperation):
    name = "end"


@irdl_op_definition
class ExternalBufferOp(IRDLOperation):
    name = "external_buffer"

    def __init__(self):
        super().__init__(result_types=[memref.MemRefType])


@irdl_op_definition
class FlowOp(IRDLOperation):
    name = "flow"

    sourceBundle: WireBundleAttr = attr_def(WireBundleAttr)
    sourceChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    destBundle: WireBundleAttr = attr_def(WireBundleAttr)
    destChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    source: Operand = operand_def(IndexType())
    dest: Operand = operand_def(IndexType())

    def __init__(
        self,
        sourceBundle: WireBundleAttr,
        sourceChannel: IntegerAttr[i32],
        destBundle: WireBundleAttr,
        destChannel: IntegerAttr[i32],
        source: IndexType(),
        dest: IndexType(),
    ):
        super().__init__(
            attributes={
                "sourceBundle": sourceBundle,
                "sourceChannel": sourceChannel,
                "destBundle": destBundle,
                "destChannel": destChannel,
            },
            operands=[source, dest],
        )


@irdl_op_definition
class GetCascadeOp(IRDLOperation):
    name = "getCascade"

    def __init__(self):
        super().__init__(result_types=[IntegerType(CASCADE_SIZE)])


@irdl_op_definition
class GetStreamOp(IRDLOperation):
    name = "getStream"

    channel: Operand = operand_def(i32)

    def __init__(self, channel: i32):
        super().__init__(
            operands=[channel],
            result_types=[
                IntegerType(32, Signedness.SIGNLESS)
                | Float32Type
                | IntegerType(128, Signedness.SIGNLESS)
            ],
        )


@irdl_op_definition
class LockOp(IRDLOperation):
    name = "lock"

    lockID: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    init: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    tile: Operand = operand_def(IndexType())

    def __init__(
        self, lockID: IntegerAttr[i32], init: IntegerAttr[i32], tile: IndexType()
    ):
        super().__init__(
            attributes={"lockID": lockID, "init": init},
            operands=[tile],
            result_types=[IndexType()],
        )


@irdl_op_definition
class MasterSetOp(IRDLOperation):
    name = "masterset"

    destBundle: WireBundleAttr = attr_def(WireBundleAttr)
    destChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    amsels: Operand = operand_def(IndexType())

    def __init__(
        self,
        destBundle: WireBundleAttr,
        destChannel: IntegerAttr[i32],
        amsels: IndexType(),
    ):
        super().__init__(
            attributes={"destBundle": destBundle, "destChannel": destChannel},
            operands=[amsels],
            result_types=[IndexType()],
        )


@irdl_op_definition
class MemOp(IRDLOperation):
    name = "mem"

    tile: Operand = operand_def(IndexType())

    def __init__(self, tile: IndexType()):
        super().__init__(operands=[tile], result_types=[IndexType()])


@irdl_op_definition
class MemTileDMAOp(IRDLOperation):
    name = "memTileDMA"

    tile: Operand = operand_def(IndexType())

    def __init__(self, tile: IndexType()):
        super().__init__(operands=[tile], result_types=[IndexType()])


""" # TODO: add successor
@irld_op_definition
class NextBDOp(IRDLOperation):
    name = "nextBd"

"""
""" # TODO objectfifotype
@irdl_op_definition
class ObjectFifoAcquireOp(IRDLOperation):
    name = "objectFifo.acquire"

    port: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(IntegerAttr[Annotated[IntegerType, i32]])
    size: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    fifo: Operand = operand_def(

@irdl_op_definition
class ObjectFifoCreateOp(IRDLOperation):
    name = "objectFifo.createObjectFifo"
"""


@irdl_op_definition
class PLIOOp(IRDLOperation):
    name = "plio"

    col: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(self, col: IntegerAttr[i32]):
        super().__init__(attributes={"col": col}, result_types=[IndexType()])


@irdl_op_definition
class PacketDestOp(IRDLOperation):
    name = "packet_dest"

    bundle: WireBundleAttr = attr_def(WireBundleAttr)
    channel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    tile: Operand = operand_def(IndexType())

    def __init__(
        self, bundle: WireBundleAttr, channel: IntegerAttr[i32], tile: IndexType()
    ):
        super().__init__(
            attributes={"bundle": bundle, "channel": channel}, operands=[tile]
        )


@irdl_op_definition
class PacketFlowOp(IRDLOperation):
    name = "packet_flow"

    ID: IntegerAttr[i8] = attr_def(IntegerAttr[i8])

    def __init__(self, ID: IntegerAttr[i8]):
        super().__init__(attributes={"ID": ID})


@irdl_op_definition
class PacketRuleOp(IRDLOperation):
    name = "rule"

    mask: IntegerAttr[i8] = attr_def(IntegerAttr[i8])
    value: IntegerAttr[i8] = attr_def(IntegerAttr[i8])

    amsel: Operand = operand_def(IndexType())

    def __init__(
        self, mask: IntegerAttr[i8], value: IntegerAttr[i8], amsel: IndexType()
    ):
        super().__init__(attributes={"mask": mask, "value": value}, operands=[amsel])


@irdl_op_definition
class PacketRulesOp(IRDLOperation):
    name = "packetrules"

    sourceBundle: WireBundleAttr = attr_def(WireBundleAttr)
    sourceChannel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(self, sourceBundle: WireBundleAttr, sourceChannel: IntegerAttr[i32]):
        super().__init__(
            attributes={"sourceBundle": sourceBundle, "sourceChannel": sourceChannel}
        )


@irdl_op_definition
class PacketSourceOp(IRDLOperation):
    name = "packet_source"

    bundle: WireBundleAttr = attr_def(WireBundleAttr)
    channel: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    tile: Operand = operand_def(IndexType())

    def __init__(
        self, bundle: WireBundleAttr, channel: IntegerAttr[i32], tile: IndexType()
    ):
        super().__init__(
            attributes={"bundle": bundle, "channel": channel}, operands=[tile]
        )


@irdl_op_definition
class putCascade(IRDLOperation):
    name = "putCascade"

    cascadeValue: Operand = operand_def(IntegerType(CASCADE_SIZE))

    def __init__(self, cascadeValue: IntegerType(CASCADE_SIZE)):
        super().__init__(operands=[cascadeValue])


@irdl_op_definition
class putStream(IRDLOperation):
    name = "putStream"

    channel: Operand = operand_def(i32)
    streamValue: Operand = operand_def(
        IntegerType(32, Signedness.SIGNLESS)
        | Float32Type
        | IntegerType(128, Signedness.SIGNLESS)
    )

    def __init__(
        self,
        channel: i32,
        streamValue: IntegerType(32, Signedness.SIGNLESS)
        | Float32Type
        | IntegerType(128, Signedness.SIGNLESS),
    ):
        super().__init__(operands=[channel, streamValue])


@irdl_op_definition
class ShimDMAAllocationOp(IRDLOperation):
    name = "shimDMAAllocation"

    sym_name: StringAttr = attr_def(StringAttr)
    channelDir: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    channelIndex: IntegerAttr[i32] = attr_def(IntegerAttr[i64])
    col: IntegerAttr[i32] = attr_def(IntegerAttr[i64])

    def __init__(
        self,
        sym_name: StringAttr,
        channelDir: IntegerAttr[i32],
        channelIndex: IntegerAttr[i32],
        col: IntegerAttr[i32],
    ):
        super().__init__(
            attributes={
                "sym_name": sym_name,
                "channelDir": channelDir,
                "channelIndex": channelIndex,
                "col": col,
            }
        )


@irdl_op_definition
class ShimDMAOp(IRDLOperation):
    name = "shimDMA"

    tile: Operand = operand_def(IndexType())

    def __init__(self, tile: IndexType()):
        super().__init__(operands=[tile], result_types=[IndexType()])


@irdl_op_definition
class ShimMuxOp(IRDLOperation):
    name = "shimmux"

    tile: Operand = operand_def(IndexType())

    def __init__(self, tile: IndexType()):
        super().__init__(operands=[tile], result_types=[IndexType()])


@irdl_op_definition
class ShimSwitchBoxOp(IRDLOperation):
    name = "shimswitchbox"

    col: IntegerAttr[i32] = attr_def(IntegerAttr[i32])

    def __init__(self, col: IntegerAttr[i32]):
        super().__init__(attributes={"col": col}, result_types=[IndexType()])


@irdl_op_definition
class SwitchboxOp(IRDLOperation):
    name = "switchbox"

    tile: Operand = operand_def(IndexType())

    def __init__(self, tile: IndexType()):
        super().__init__(self, operands=[tile], result_types=[tile])


@irdl_op_definition
class UseLockOp(IRDLOperation):
    name = "useLock"

    value: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    action: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    blocking: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )

    def __init__(
        self,
        value: IntegerAttr[i32],
        action: IntegerAttr[i32],
        blocking: IntegerAttr[i32],
        lock: IndexType(),
    ):
        super().__init__(
            attributes={"value": value, "action": action, "blocking": blocking},
            operands=[lock],
        )


@irdl_op_definition
class WireOp(IRDLOperation):
    name = "wire"

    sourceBundle: WireBundleAttr = attr_def(WireBundleAttr)
    destBundle: WireBundleAttr = attr_def(WireBundleAttr)
    source: Operand = operand_def(IndexType())
    dest: Operand = operand_def(IndexType())

    def __init__(
        self,
        sourceBundle: WireBundleAttr,
        destBundle: WireBundleAttr,
        source: IndexType(),
        dest: IndexType(),
    ):
        super().__init__(
            attributes={"sourceBundle": sourceBundle, "destBundle": destBundle},
            operands=[source, dest],
        )


AIE = Dialect([AMSelOp, BufferOp, TileOp, ConnectOp, CoreOp, DMAStartOp], [])
