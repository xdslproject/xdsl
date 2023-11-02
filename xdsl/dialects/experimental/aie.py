from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, TypeVar

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyAttr,
    AnyIntegerAttr,
    ArrayAttr,
    FlatSymbolRefAttr,
    Float32Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    NoneAttr,
    Region,
    Signedness,
    StringAttr,
    SymbolRefAttr,
    i32,
    i64,
)
from xdsl.ir import Attribute, Data, Dialect, OpResult, ParametrizedAttribute
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    ParameterDef,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import SymbolOpInterface, SymbolTable

_MemRefTypeElement = TypeVar("_MemRefTypeElement", bound=Attribute)


i8 = IntegerType(8)

CASCADE_SIZE = 384

LOCK_ACQUIRE = 1
LOCK_RELEASE = 0

# @irdl_attr_definition
# class DimTupleArrayAttr():
#    @classmethod
#    def parse_parameter


@irdl_attr_definition
class WireBundleAttr(Data[str]):
    name = "wire_bundle"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')


@irdl_attr_definition
class ObjectFIFO(ParametrizedAttribute):
    name = "AIE.objectFifo"

    buffer: ParameterDef[memref.MemRefType]

    @staticmethod
    def from_element_type_and_shape(
        referenced_type: _MemRefTypeElement,
        shape: Iterable[int | AnyIntegerAttr],
        layout: Attribute = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> ObjectFIFO[_MemRefTypeElement]:
        return ObjectFIFO(
            [
                memref.MemRefType.from_element_type_and_shape(
                    referenced_type, shape, layout, memory_space
                )
            ]
        )


@irdl_attr_definition
class ObjectFIFOSubview(ParametrizedAttribute):
    name = "AIE.objectFifoSubview"

    buffer: ParameterDef[memref.MemRefType]

    @staticmethod
    def from_element_type_and_shape(
        object_fifo: ObjectFIFO,
        shape: Iterable[int | AnyIntegerAttr],
    ) -> ObjectFIFOSubview[_MemRefTypeElement]:
        return ObjectFIFOSubview(
            [
                memref.MemRefType.from_element_type_and_shape(
                    object_fifo.buffer.referenced_type,
                    shape,
                    object_fifo.buffer.layout,
                    object_fifo.buffer.memory_space,
                )
            ]
        )


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
    name = "AIE.buffer"
    tile: Operand = operand_def(IndexType())
    shape: ArrayAttr[i32] = attr_def(ArrayAttr[i32])
    element_type: AnyAttr = attr_def(AnyAttr())
    buffer: OpResult = result_def(memref.MemRefType)
    sym_name: StringAttr = attr_def(StringAttr)

    def __init__(
        self,
        tile: IndexType(),
        element_type: AnyAttr,
        shape: ArrayAttr[i32],
        sym_name: StringAttr,
    ):
        buffer_type = memref.MemRefType.from_element_type_and_shape(element_type, shape)
        super().__init__(
            operands=[tile],
            attributes={"sym_name": sym_name},
            result_types=[buffer_type],
        )


@irdl_op_definition
class TileOp(IRDLOperation):
    name = "AIE.tile"
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
    name = "AIE.core"
    stackSize: IntegerAttr[i32] = attr_def(IntegerAttr)
    tile: Operand = operand_def(IndexType())
    region: Region = region_def()
    result: OpResult = result_def(IndexType())

    def __init__(self, stackSize: IntegerAttr[i32], tile: IndexType(), region: Region):
        super().__init__(
            attributes={"stackSize": stackSize},
            operands=[tile],
            regions=[region],
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
    arg: Operand = operand_def(AnyAttr())

    def __init__(self, arg: AnyAttr()):
        super().__init__(operands=[arg])


@irdl_op_definition
class DeviceOp(IRDLOperation):
    name = "AIE.device"

    region: Region = region_def()

    device: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )

    def __init__(self, device: IntegerAttr[i32], region: Region):
        super().__init__(attributes={"device": device}, regions=[region])

    def print(self, printer: Printer):
        printer.print("(")
        device_str = "xcvc1902" if self.device.value.data == 0 else ""
        printer.print(device_str)
        printer.print(") ")
        printer.print_region(self.region)


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
    name = "AIE.lock"

    lockID: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    init: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    tile: Operand = operand_def(IndexType())
    result: OpResult = result_def(IndexType())
    sym_name: StringAttr = attr_def(StringAttr)

    def __init__(
        self,
        lockID: IntegerAttr[i32],
        init: IntegerAttr[i32],
        tile: IndexType(),
        sym_name: StringAttr,
    ):
        super().__init__(
            attributes={"lockID": lockID, "init": init, "sym_name": sym_name},
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


# @irdl_op_definition
# class NextBDOp(IRDLOperation):
#    name = "nextBd"
#
#    dest: Block
#
#    def __init__(self, dest: Block):
#        super().__init__(successors=[dest])


@irdl_op_definition
class ObjectFifoAcquireOp(IRDLOperation):
    name = "AIE.objectFifo.acquire"

    port: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    size: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    object_fifo: FlatSymbolRefAttr = prop_def(FlatSymbolRefAttr)

    result: OpResult = result_def(i32)

    def __init__(
        self,
        port: IntegerAttr[i32],
        size: IntegerAttr[i32],
        object_fifo: str | SymbolRefAttr,
    ):
        if isinstance(object_fifo, str):
            object_fifo = StringAttr(object_fifo)
        super().__init__(
            attributes={"port": port, "size": size},
            properties={"object_fifo": object_fifo},
            result_types=[i32],
        )

    def print(self, printer: Printer):
        port = "Produce" if self.port == 0 else "Consume"
        op = self.parent_op()
        while not isinstance(op, ModuleOp):
            op = op.parent_op()

        print("TYPE (module?): ", type(op))
        printer.print(f" @{self.object_fifo.data} (", port, ", ", self.size, ")")
        print("FIFO_DATA: ", self.object_fifo.data)
        lookup = SymbolTable.lookup_symbol(op, SymbolRefAttr("of", ["object_fifo"]))
        print(f"LOOKUP: {lookup}")


@irdl_op_definition
class createObjectFifo(IRDLOperation):
    name = "AIE.objectFifo"

    elemNumber: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    producerTile: Operand = operand_def(IndexType())
    consumerTile: Operand = operand_def(IndexType())
    sym_name: StringAttr = prop_def(StringAttr)
    object_fifo: ObjectFIFO = attr_def(ObjectFIFO)

    traits = frozenset([SymbolOpInterface()])

    def __init__(
        self,
        elemNumber: IntegerAttr[i32],
        producerTile: IndexType(),
        consumerTile: IndexType(),
        referenced_type: _MemRefTypeElement,
        shape: Iterable[int | AnyIntegerAttr],
        name: str,
    ):
        object_fifo = ObjectFIFO.from_element_type_and_shape(referenced_type, shape)
        super().__init__(
            attributes={"elemNumber": elemNumber, "object_fifo": object_fifo},
            properties={"sym_name": StringAttr(name)},
            operands=[producerTile, consumerTile],
        )

    def print(self, printer: Printer):
        printer.print(
            " @",
            self.sym_name.data,
            "(",
            self.producerTile,
            ", {",
            self.consumerTile,
            "}) : !AIE.objectFIFO<",
            self.object_fifo.buffer,
            ">",
        )


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
        or Float32Type
        or IntegerType(128, Signedness.SIGNLESS)
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
    name = "AIE.useLock"

    value: IntegerAttr[i32] = attr_def(IntegerAttr[i32])
    action: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    blocking: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    lock: Operand = operand_def(IndexType())

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

    def print(self, printer: Printer):
        printer.print("(")
        printer.print_operand(self.lock)
        action_str = (
            '"Acquire"' if self.action.value.data == LOCK_ACQUIRE else '"Release"'
        )
        printer.print(", ")
        printer.print(action_str)
        printer.print(", ")
        printer.print(self.value.value.data)
        printer.print(")")


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


@irdl_op_definition
class EndOp(IRDLOperation):
    name = "AIE.end"

    def __init__(self):
        super().__init__()


AIE = Dialect([AMSelOp, BufferOp, TileOp, ConnectOp, CoreOp, DMAStartOp], [])
