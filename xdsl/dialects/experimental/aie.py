"""
Port of the AMD Xilinx AIE dialect for programming the AIEs on the AMD Xilinx Versal FPGA architecture.
AIE is a hardened systolic array present in the Versal devices. The dialect describes netlists of AIE
components and it can be lowered to the processor's assembly using the vendor's compiler. A description
of the original dialect can be found here https://xilinx.github.io/mlir-aie/AIEDialect.html
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import auto
from typing import Annotated, Generic

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyAttr,
    AnyIntegerAttr,
    ArrayAttr,
    Block,
    FlatSymbolRefAttr,
    Float32Type,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    Region,
    Signedness,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    AttributeInvT,
    Data,
    Dialect,
    EnumAttribute,
    Operation,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    StrEnum,
)
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    successor_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator, SymbolOpInterface, SymbolTable
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

i8 = IntegerType(8)

CASCADE_SIZE = 384

LOCK_ACQUIRE = 1
LOCK_RELEASE = 0

PRODUCE_PORT = 0
CONSUME_PORT = 1

BLOCKING = 1
NONBLOCKING = 0

MM2S = 0
S2MM = 1


class AIEDeviceEnum(StrEnum):
    xcvc1902 = auto()


@irdl_attr_definition
class AIEDeviceAttr(EnumAttribute[AIEDeviceEnum]):
    name = "aie.device_attr"


@irdl_attr_definition
class WireBundleAttr(Data[str]):
    name = "aie.wire_bundle"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)


@irdl_attr_definition
class ObjectFIFO(Generic[AttributeInvT], ParametrizedAttribute):
    name = "aie.objectFifo"

    buffer: ParameterDef[AttributeInvT]

    @staticmethod
    def from_element_type_and_shape(
        referenced_type: AttributeInvT,
        shape: Iterable[int | IntAttr],
        layout: Attribute = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ) -> ObjectFIFO[AttributeInvT]:
        return ObjectFIFO(
            [memref.MemRefType(referenced_type, shape, layout, memory_space)]
        )


@irdl_attr_definition
class ObjectFIFOSubview(Generic[AttributeInvT], ParametrizedAttribute):
    name = "aie.objectFifoSubview"

    buffer: ParameterDef[AttributeInvT]

    @staticmethod
    def from_element_type_and_shape(
        object_fifo: ObjectFIFO[AttributeInvT],
        shape: Iterable[int | IntAttr],
    ) -> ObjectFIFOSubview[AttributeInvT]:
        assert isa(object_fifo.buffer, memref.MemRefType[Attribute])
        return ObjectFIFOSubview(
            [
                memref.MemRefType(
                    object_fifo.buffer.element_type,
                    shape,
                    object_fifo.buffer.layout,
                    object_fifo.buffer.memory_space,
                )
            ]
        )


@irdl_op_definition
class AMSelOp(IRDLOperation):
    name = "aie.amsel"
    arbiterID = attr_def(AnyIntegerAttr)
    msel = attr_def(AnyIntegerAttr)
    result = result_def(IndexType())

    def __init__(
        self, arbiterID: IntegerAttr[IntegerType], msel: IntegerAttr[IntegerType]
    ):
        super().__init__(
            attributes={"arbiterID": arbiterID, "msel": msel},
            result_types=[IndexType()],
        )

    def verify_(self) -> None:
        if self.arbiterID.type != IntegerType(8):
            raise VerifyException("arbiterID has to be an 8-bit signless integer")
        if self.arbiterID.value.data < 0 or self.arbiterID.value.data > 5:
            raise VerifyException("arbiterID has to be in the range [0-5].")
        if self.msel.type != IntegerType(8):
            raise VerifyException("msel has to be an 8-bit signless integer")
        if self.msel.value.data < 0 or self.arbiterID.value.data > 3:
            raise VerifyException("msel has to be in the range [0-3].")


@irdl_op_definition
class BufferOp(IRDLOperation):
    name = "aie.buffer"
    tile = operand_def(IndexType())
    # shape = attr_def(ArrayAttr[IntegerType])
    # element_type = attr_def(Attribute)
    buffer = result_def(memref.MemRefType)
    sym_name = attr_def(StringAttr)

    def __init__(
        self,
        tile: Operation | SSAValue,
        element_type: Attribute,
        shape: ArrayAttr[AnyIntegerAttr],
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
    name = "aie.tile"
    col = attr_def(IntegerAttr[IntegerType])
    row = attr_def(IntegerAttr[IntegerType])
    result = result_def(IndexType())

    def __init__(self, col: IntegerAttr[IntegerType], row: IntegerAttr[IntegerType]):
        super().__init__(
            attributes={"col": col, "row": row}, result_types=[IndexType()]
        )


@irdl_op_definition
class ConnectOp(IRDLOperation):
    name = "aie.connect"
    sourceBundle = attr_def(WireBundleAttr)
    sourceChannel = attr_def(AnyIntegerAttr)
    destBundle = attr_def(WireBundleAttr)
    destChannel = attr_def(AnyIntegerAttr)

    def __init__(
        self,
        sourceBundle: WireBundleAttr,
        sourceChannel: int | AnyIntegerAttr,
        destBundle: WireBundleAttr,
        destChannel: int | AnyIntegerAttr,
    ):
        if isinstance(sourceChannel, int):
            sourceChannel = IntegerAttr(sourceChannel, IntegerType(8))
        if isinstance(destChannel, int):
            destChannel = IntegerAttr(destChannel, IntegerType(8))
        super().__init__(
            attributes={
                "sourceBundle": sourceBundle,
                "sourceChannel": sourceChannel,
                "destBundle": destBundle,
                "destChannel": destChannel,
            }
        )

    def print(self, printer: Printer):
        printer.print(
            "<",
            self.sourceBundle.data,
            ": ",
            self.sourceChannel.value.data,
            ", ",
            self.destBundle.data,
            ": ",
            self.destChannel.value.data,
            ">",
        )

    def verify_(self) -> None:
        if self.sourceChannel.type != i32:
            raise VerifyException("sourceChannel has to be a 32-bit signless integer")
        if self.sourceChannel.value.data < 0:
            raise VerifyException("sourceChannel has to be equal or greater than 0")
        if self.destChannel.type != i32:
            raise VerifyException("destChannel has to be a 32-bit signless integer")
        if self.destChannel.value.data < 0:
            raise VerifyException("destChannel has to be equal or greater than 0")


@irdl_op_definition
class CoreOp(IRDLOperation):
    name = "aie.core"
    stackSize = opt_attr_def(IntegerType)
    tile = operand_def(IndexType())
    region = region_def()
    link_with = opt_attr_def(StringAttr)
    result = result_def(IndexType())

    def __init__(
        self,
        stackSize: IntegerAttr[IntegerType] | None,
        tile: Operation | SSAValue,
        region: Region,
        link_with: StringAttr | None = None,
    ):
        super().__init__(
            attributes={"stackSize": stackSize, "link_with": link_with},
            operands=[tile],
            regions=[region],
            result_types=[IndexType()],
        )

    def print(self, printer: Printer):
        printer.print(" (")
        printer.print_operand(self.tile)
        printer.print(") ")
        printer.print_region(self.region)
        if self.link_with is not None:
            printer.print(' { link_with="', self.link_with, '" }')

    @classmethod
    def parse(cls, parser: Parser) -> CoreOp:
        parser.parse_characters("(")
        tile = parser.parse_operand()
        parser.parse_characters(")")
        region = parser.parse_region()

        stackSize = None
        return CoreOp(stackSize, tile, region)


@irdl_op_definition
class DMABDOp(IRDLOperation):
    name = "aie.dmaBd"
    offset = attr_def(IntegerAttr[IntegerType])
    length = attr_def(IntegerAttr[IntegerType])
    ab = attr_def(IntegerAttr[IntegerType], attr_name="AB")
    buffer = operand_def(memref.MemRefType)
    dimensions = opt_attr_def(
        IntegerAttr[IntegerType]
    )  # TODO: this should be implemented as a DimTupleArrayAttr: check https://xilinx.github.io/mlir-aie/AIEDialect.html

    def __init__(
        self,
        offset: IntegerAttr[IntegerType],
        length: IntegerAttr[IntegerType],
        AB: IntegerAttr[IntegerType],
        dimensions: None | IntegerAttr[IntegerType],
        buffer: Operation | SSAValue,
    ):
        if dimensions is None:
            super().__init__(
                attributes={"offset": offset, "length": length, "AB": AB},
                operands=[buffer],
            )
        else:
            super().__init__(
                attributes={
                    "offset": offset,
                    "length": length,
                    "AB": AB,
                    "dimensions": dimensions,
                },
                operands=[buffer],
            )

    def print(self, printer: Printer):
        printer.print("(<")
        printer.print_operand(self.buffer)
        printer.print(
            ": ",
            self.buffer.type,
            ", ",
            self.offset.value.data,
            ", ",
            self.length.value.data,
            ">, ",
            self.ab.value.data,
            ")",
        )


@irdl_op_definition
class DMABDPACKETOp(IRDLOperation):
    name = "aie.dmaBdPacket"
    packet_type = attr_def(IntegerAttr[IntegerType])
    packet_id = attr_def(IntegerAttr[IntegerType])

    def __init__(
        self, packet_type: IntegerAttr[IntegerType], packet_id: IntegerAttr[IntegerType]
    ):
        super().__init__(
            attributes={"packet_type": packet_type, "packet_id": packet_id}
        )

    def print(self, printer: Printer):
        printer.print(
            "(",
            f"0x{self.packet_type.value.data:X}",
            ", ",
            f"0x{self.packet_id.value.data:X}",
            ")",
        )


@irdl_op_definition
class DMAStartOp(IRDLOperation):
    name = "aie.dmaStart"
    channelDir = attr_def(IntegerAttr[Annotated[IntegerType, i32]])
    channelIndex = attr_def(IntegerAttr[IntegerType])
    dest = successor_def()
    chain = successor_def()

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        channelDir: IntegerAttr[IntegerType],
        channelIndex: IntegerAttr[IntegerType],
        dest: Block,
        chain: Block,
    ):
        super().__init__(
            attributes={"channelDir": channelDir, "channelIndex": channelIndex},
            successors=[dest, chain],
        )

    def print(self, printer: Printer):
        direction = "MM2S" if self.channelDir.value.data == 0 else "S2MM"
        printer.print("(", direction, ", ", self.channelIndex.value.data, ", ")
        printer.print_block_name(self.dest)
        printer.print(", ")
        printer.print_block_name(self.chain)
        printer.print(")")


@irdl_op_definition
class DebugOp(IRDLOperation):
    name = "aie.debug"
    arg = operand_def(AnyAttr())

    def __init__(self, arg: Operation | SSAValue):
        super().__init__(operands=[arg])


@irdl_op_definition
class DeviceOp(IRDLOperation):
    name = "aie.device"

    region = region_def()

    device = attr_def(AIEDeviceAttr)
    traits = frozenset([SymbolTable()])

    def __init__(self, device: IntegerAttr[IntegerType], region: Region):
        super().__init__(attributes={"device": device}, regions=[region])

    assembly_format = "$device attr-dict"


@irdl_op_definition
class ExternalBufferOp(IRDLOperation):
    name = "aie.external_buffer"

    sym_name = attr_def(StringAttr)
    buffer = result_def(memref.MemRefType)

    def __init__(
        self,
        sym_name: str,
        shape: ArrayAttr[AnyIntegerAttr],
        element_type: Attribute,
    ):
        super().__init__(
            attributes={"sym_name": StringAttr(sym_name)},
            result_types=[
                memref.MemRefType.from_element_type_and_shape(element_type, shape)
            ],
        )


@irdl_op_definition
class FlowOp(IRDLOperation):
    name = "aie.flow"

    sourceBundle = attr_def(WireBundleAttr)
    sourceChannel = attr_def(IntegerAttr[IntegerType])
    destBundle = attr_def(WireBundleAttr)
    destChannel = attr_def(IntegerAttr[IntegerType])
    source = operand_def(IndexType())
    dest = operand_def(IndexType())

    def __init__(
        self,
        sourceBundle: WireBundleAttr,
        sourceChannel: IntegerAttr[IntegerType],
        destBundle: WireBundleAttr,
        destChannel: IntegerAttr[IntegerType],
        source: Operation | SSAValue,
        dest: Operation | SSAValue,
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

    assembly_format = "`(` $source `, ` $dest `)` `<` $sourceBundle `: ` $sourceChannel `, ` $destBundle `: ` $destChannel `>`  attr-dict"


@irdl_op_definition
class GetCascadeOp(IRDLOperation):
    name = "getCascade"

    def __init__(self):
        super().__init__(result_types=[IntegerType(CASCADE_SIZE)])


@irdl_op_definition
class GetStreamOp(IRDLOperation):
    name = "aie.getStream"

    channel = operand_def(i32)
    result = result_def(IntegerType)

    def __init__(self, channel: Operation | SSAValue):
        super().__init__(
            operands=[channel],
            result_types=[i32],  # TODO: the result can be of several types.
        )


@irdl_op_definition
class LockOp(IRDLOperation):
    name = "aie.lock"

    lockID = attr_def(IntegerAttr[IntegerType])
    init = attr_def(IntegerAttr[IntegerType])
    tile = operand_def(IndexType())
    result = result_def(IndexType())
    sym_name = attr_def(StringAttr)

    def __init__(
        self,
        lockID: IntegerAttr[IntegerType],
        init: IntegerAttr[IntegerType],
        tile: Operation | SSAValue,
        sym_name: StringAttr,
    ):
        super().__init__(
            attributes={"lockID": lockID, "init": init, "sym_name": sym_name},
            operands=[tile],
            result_types=[IndexType()],
        )


@irdl_op_definition
class MasterSetOp(IRDLOperation):
    name = "aie.masterset"

    destBundle = attr_def(WireBundleAttr)
    destChannel = attr_def(IntegerAttr[IntegerType])
    amsels = operand_def(IndexType())

    def __init__(
        self,
        destBundle: WireBundleAttr,
        destChannel: IntegerAttr[IntegerType],
        amsels: Operation | SSAValue,
    ):
        super().__init__(
            attributes={"destBundle": destBundle, "destChannel": destChannel},
            operands=[amsels],
            result_types=[IndexType()],
        )


@irdl_op_definition
class MemOp(IRDLOperation):
    name = "aie.mem"

    tile = operand_def(IndexType())
    region = region_def()
    result = result_def(IndexType())

    def __init__(self, tile: Operation | SSAValue, region: Region):
        super().__init__(operands=[tile], result_types=[IndexType()], regions=[region])


@irdl_op_definition
class MemTileDMAOp(IRDLOperation):
    name = "aie.memTileDMA"

    tile = operand_def(IndexType())

    def __init__(self, tile: Operation | SSAValue):
        super().__init__(operands=[tile], result_types=[IndexType()])


@irdl_op_definition
class NextBDOp(IRDLOperation):
    name = "aie.nextBd"

    dest = successor_def()

    def __init__(self, dest: Block):
        super().__init__(successors=[dest])


@irdl_op_definition
class ObjectFifoAcquireOp(IRDLOperation):
    name = "aie.objectFifo.acquire"

    port = attr_def(IntegerAttr[Annotated[IntegerType, i32]])
    size = attr_def(IntegerAttr[IntegerType])
    object_fifo = attr_def(FlatSymbolRefAttr)

    result: OpResult = result_def(ObjectFIFOSubview)

    def __init__(
        self,
        port: IntegerAttr[IntegerType],
        size: IntegerAttr[IntegerType],
        object_fifo: str | SymbolRefAttr,
        device: DeviceOp,
    ):
        if isinstance(object_fifo, str):
            object_fifo = SymbolRefAttr(object_fifo)

        lookup = SymbolTable.lookup_symbol(device, object_fifo)
        assert isinstance(lookup, createObjectFifo)
        assert isa(lookup.object_fifo, ObjectFIFO[memref.MemRefType[Attribute]])
        subview_shape = [
            shape_elem.data for shape_elem in lookup.object_fifo.buffer.shape.data
        ]
        result_subview = ObjectFIFOSubview.from_element_type_and_shape(
            lookup.object_fifo, subview_shape
        )
        super().__init__(
            attributes={"port": port, "size": size, "object_fifo": object_fifo},
            result_types=[result_subview],
        )

    def print(self, printer: Printer):
        port = "Produce" if self.port == PRODUCE_PORT else "Consume"
        op = self.parent_op()
        while not isinstance(op, DeviceOp):
            assert isinstance(op, Operation)
            op = op.parent_op()

        assert isinstance(self.object_fifo, StringAttr)
        printer.print(f" @{self.object_fifo.data} (", port, ", ", self.size, ") : ")
        lookup = SymbolTable.lookup_symbol(op, self.object_fifo)
        assert isinstance(lookup, createObjectFifo)
        assert isa(lookup.object_fifo, ObjectFIFO[memref.MemRefType[Attribute]])
        printer.print("!AIE.objectFifoSubview<", lookup.object_fifo.buffer, ">")


@irdl_op_definition
class ObjectFifoRegisterExternalBuffersOp(IRDLOperation):
    name = "aie.objectFifo.registerExternalBuffers"

    tile = operand_def(IndexType())
    externalBuffers = operand_def(memref.MemRefType)
    object_fifo = attr_def(FlatSymbolRefAttr)

    def __init__(
        self,
        tile: Operation | SSAValue,
        externalBuffers: Operation | SSAValue,
        object_fifo: str | SymbolRefAttr,
    ):
        if isinstance(object_fifo, str):
            object_fifo = SymbolRefAttr(object_fifo)

        super().__init__(
            operands=[tile, externalBuffers], attributes={"object_fifo": object_fifo}
        )

    def print(self, printer: Printer):
        assert isinstance(self.object_fifo, StringAttr)
        printer.print(
            f" @{self.object_fifo.data} (",
            self.tile,
            ", {",
            self.externalBuffers,
            "}) : (",
            self.externalBuffers.type,
            ")",
        )


@irdl_op_definition
class ObjectFIFOSubviewAccessOp(IRDLOperation):
    name = "aie.objectFifo.subview.access"

    index = attr_def(IntegerAttr[IntegerType])
    subview = operand_def(ObjectFIFOSubview[memref.MemRefType[Attribute]])
    output = result_def(memref.MemRefType)

    def __init__(self, index: IntegerAttr[IntegerType], subview: Operation | SSAValue):
        assert isinstance(subview, ObjectFifoAcquireOp)
        assert isa(subview.result.type, ObjectFIFOSubview[memref.MemRefType[Attribute]])
        subview.result.type.buffer
        result_type = memref.MemRefType(
            subview.result.type.buffer.element_type, subview.result.type.buffer.shape
        )
        super().__init__(
            attributes={"index": index}, operands=[subview], result_types=[result_type]
        )

    def print(self, printer: Printer):
        assert isa(self.subview.type, ObjectFIFOSubview[memref.MemRefType[Attribute]])
        printer.print(" ")
        printer.print_operand(self.subview)
        printer.print("[", self.index.value.data, "] : ")
        printer.print(
            "!AIE.objectFifoSubview<",
            self.subview.type.buffer,
            "> -> ",
            self.subview.type.buffer,
        )


@irdl_op_definition
class createObjectFifo(IRDLOperation):
    name = "aie.objectFifo"

    elemNumber = attr_def(IntegerAttr[IntegerType])
    producerTile = operand_def(IndexType())
    consumerTile = var_operand_def(IndexType())
    sym_name = attr_def(StringAttr)
    object_fifo = attr_def(ObjectFIFO[Attribute])

    traits = frozenset([SymbolOpInterface()])

    def __init__(
        self,
        elemNumber: IntegerAttr[IntegerType],
        producerTile: Operation | SSAValue,
        consumerTile: Operation | SSAValue,
        referenced_type: Attribute,
        shape: Iterable[int | IntAttr],
        name: str,
    ):
        object_fifo = ObjectFIFO.from_element_type_and_shape(referenced_type, shape)
        super().__init__(
            attributes={
                "elemNumber": elemNumber,
                "object_fifo": object_fifo,
                "sym_name": StringAttr(name),
            },
            operands=[producerTile, consumerTile],
        )

    def print(self, printer: Printer):
        printer.print(" @", self.sym_name.data, "(", self.producerTile, ", {")
        for i in range(len(self.consumerTile) - 1):
            printer.print(self.consumerTile[i], ", ")

        printer.print(self.consumerTile[-1], "}, ")
        printer.print_attribute(self.elemNumber)

        printer.print(
            ") : !AIE.objectFifo<",
            self.object_fifo.buffer,
            ">",
        )


@irdl_op_definition
class ObjectFIFOReleaseOp(IRDLOperation):
    name = "aie.objectFifo.release"

    port = attr_def(IntegerAttr[IntegerType])
    size = attr_def(IntegerAttr[IntegerType])
    object_fifo = attr_def(FlatSymbolRefAttr)

    def __init__(
        self,
        port: IntegerAttr[IntegerType],
        size: IntegerAttr[IntegerType],
        object_fifo: str | SymbolRefAttr,
    ):
        if isinstance(object_fifo, str):
            object_fifo = SymbolRefAttr(object_fifo)

        super().__init__(
            attributes={"port": port, "size": size, "object_fifo": object_fifo}
        )

    def print(self, printer: Printer):
        assert isinstance(self.object_fifo, StringAttr)
        if isinstance(self.object_fifo, str):
            ref_objectfifo = self.object_fifo
        else:
            ref_objectfifo = self.object_fifo.data

        printer.print(
            " @",
            ref_objectfifo,
            " (",
            "Produce" if self.port.value.data == PRODUCE_PORT else "Consume",
            ", ",
            self.size.value.data,
            ")",
        )


@irdl_op_definition
class PLIOOp(IRDLOperation):
    name = "aie.plio"

    col = attr_def(IntegerAttr[IntegerType])

    def __init__(self, col: IntegerAttr[IntegerType]):
        super().__init__(attributes={"col": col}, result_types=[IndexType()])


@irdl_op_definition
class PacketDestOp(IRDLOperation):
    name = "aie.packet_dest"

    bundle = attr_def(WireBundleAttr)
    channel = attr_def(IntegerAttr[IntegerType])

    tile = operand_def(IndexType())

    def __init__(
        self,
        bundle: WireBundleAttr,
        channel: IntegerAttr[IntegerType],
        tile: Operation | SSAValue,
    ):
        super().__init__(
            attributes={"bundle": bundle, "channel": channel}, operands=[tile]
        )

    def print(self, printer: Printer):
        printer.print(
            "<", self.tile, ", ", self.bundle.data, " : ", self.channel.value.data, ">"
        )


@irdl_op_definition
class PacketFlowOp(IRDLOperation):
    name = "aie.packet_flow"

    ID = attr_def(IntegerAttr[IntegerType])
    region = region_def()

    def __init__(self, ID: IntegerAttr[IntegerType], region: Region):
        super().__init__(attributes={"ID": ID}, regions=[region])

    def print(self, printer: Printer):
        printer.print("(", f"0x{self.ID.value.data:X}", ") ")
        printer.print_region(self.region)


@irdl_op_definition
class PacketRuleOp(IRDLOperation):
    name = "aie.rule"

    mask = attr_def(IntegerAttr[IntegerType])
    value = attr_def(IntegerAttr[IntegerType])

    amsel = operand_def(IndexType())

    def __init__(
        self,
        mask: IntegerAttr[IntegerType],
        value: IntegerAttr[IntegerType],
        amsel: Operation | SSAValue,
    ):
        super().__init__(attributes={"mask": mask, "value": value}, operands=[amsel])


@irdl_op_definition
class PacketRulesOp(IRDLOperation):
    name = "aie.packetrules"

    sourceBundle = attr_def(WireBundleAttr)
    sourceChannel = attr_def(IntegerAttr[IntegerType])

    def __init__(
        self, sourceBundle: WireBundleAttr, sourceChannel: IntegerAttr[IntegerType]
    ):
        super().__init__(
            attributes={"sourceBundle": sourceBundle, "sourceChannel": sourceChannel}
        )


@irdl_op_definition
class PacketSourceOp(IRDLOperation):
    name = "aie.packet_source"

    bundle = attr_def(WireBundleAttr)
    channel = attr_def(IntegerAttr[IntegerType])
    tile = operand_def(IndexType())

    def __init__(
        self,
        bundle: WireBundleAttr,
        channel: IntegerAttr[IntegerType],
        tile: Operation | SSAValue,
    ):
        super().__init__(
            attributes={"bundle": bundle, "channel": channel}, operands=[tile]
        )

    def print(self, printer: Printer):
        printer.print(
            "<", self.tile, ", ", self.bundle.data, " : ", self.channel.value.data, ">"
        )


@irdl_op_definition
class PutCascade(IRDLOperation):
    name = "aie.putCascade"

    cascadeValue = operand_def(IntegerType(CASCADE_SIZE))

    def __init__(self, cascadeValue: Operation | SSAValue):
        super().__init__(operands=[cascadeValue])


@irdl_op_definition
class PutStream(IRDLOperation):
    name = "aie.putStream"

    channel = operand_def(i32)
    streamValue = operand_def(
        IntegerType(32, Signedness.SIGNLESS)
        or Float32Type
        or IntegerType(128, Signedness.SIGNLESS)
    )

    def __init__(
        self, channel: Operation | SSAValue, streamValue: Operation | SSAValue
    ):
        super().__init__(operands=[channel, streamValue])


@irdl_op_definition
class ShimDMAAllocationOp(IRDLOperation):
    name = "aie.shimDMAAllocation"

    sym_name = attr_def(StringAttr)
    channelDir = attr_def(IntegerAttr[Annotated[IntegerType, i32]])
    channelIndex = attr_def(IntegerAttr[IntegerType])
    col = attr_def(IntegerAttr[IntegerType])

    def __init__(
        self,
        sym_name: StringAttr,
        channelDir: IntegerAttr[IntegerType],
        channelIndex: IntegerAttr[IntegerType],
        col: IntegerAttr[IntegerType],
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
    name = "aie.shimDMA"

    tile = operand_def(IndexType())

    def __init__(self, tile: Operation | SSAValue):
        super().__init__(operands=[tile], result_types=[IndexType()])


@irdl_op_definition
class ShimMuxOp(IRDLOperation):
    name = "aie.shimmux"

    tile = operand_def(IndexType())

    def __init__(self, tile: Operation | SSAValue):
        super().__init__(operands=[tile], result_types=[IndexType()])


@irdl_op_definition
class ShimSwitchBoxOp(IRDLOperation):
    name = "aie.shimswitchbox"

    col = attr_def(IntegerAttr[IntegerType])

    def __init__(self, col: IntegerAttr[IntegerType]):
        super().__init__(attributes={"col": col}, result_types=[IndexType()])


@irdl_op_definition
class SwitchboxOp(IRDLOperation):
    name = "aie.switchbox"

    tile = operand_def(IndexType())
    region = region_def()
    result = result_def(IndexType())

    def __init__(self, tile: Operation | SSAValue, region: Region):
        super().__init__(operands=[tile], regions=[region], result_types=[IndexType()])

    def print(self, printer: Printer):
        printer.print("(")
        printer.print_operand(self.tile)
        printer.print(") ")
        printer.print_region(self.region)


@irdl_op_definition
class UseLockOp(IRDLOperation):
    name = "aie.useLock"

    value = attr_def(IntegerAttr[IntegerType])
    action = attr_def(IntegerAttr[Annotated[IntegerType, i32]])
    blocking = attr_def(IntegerAttr[Annotated[IntegerType, i32]])
    lock = operand_def(IndexType())

    def __init__(
        self,
        value: IntegerAttr[IntegerType],
        action: IntegerAttr[IntegerType],
        blocking: IntegerAttr[IntegerType],
        lock: Operation | SSAValue,
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

    @classmethod
    def parse(cls, parser: Parser) -> UseLockOp:
        parser.parse_characters("(")
        lock = parser.parse_operand()
        parser.parse_characters(",")
        action = parser.parse_str_literal()
        action = (
            IntegerAttr.from_int_and_width(LOCK_ACQUIRE, 32)
            if action == "Acquire"
            else IntegerAttr.from_int_and_width(LOCK_RELEASE, 32)
        )
        parser.parse_characters(",")
        value = IntegerAttr.from_int_and_width(parser.parse_integer(), 32)
        parser.parse_characters(")")

        blocking = IntegerAttr.from_int_and_width(BLOCKING, 32)

        return UseLockOp(value, action, blocking, lock)


@irdl_op_definition
class WireOp(IRDLOperation):
    name = "aie.wire"

    sourceBundle = attr_def(WireBundleAttr)
    destBundle = attr_def(WireBundleAttr)
    source = operand_def(IndexType())
    dest = operand_def(IndexType())

    def __init__(
        self,
        sourceBundle: WireBundleAttr,
        destBundle: WireBundleAttr,
        source: Operation | SSAValue,
        dest: Operation | SSAValue,
    ):
        super().__init__(
            attributes={"sourceBundle": sourceBundle, "destBundle": destBundle},
            operands=[source, dest],
        )


@irdl_op_definition
class EndOp(IRDLOperation):
    name = "aie.end"

    def __init__(self):
        super().__init__()

    traits = frozenset([IsTerminator()])


AIE = Dialect(
    "aie",
    [
        AMSelOp,
        BufferOp,
        TileOp,
        ConnectOp,
        CoreOp,
        DMABDOp,
        DMABDPACKETOp,
        DMAStartOp,
        DebugOp,
        DeviceOp,
        ExternalBufferOp,
        FlowOp,
        GetCascadeOp,
        GetStreamOp,
        LockOp,
        MasterSetOp,
        MemOp,
        MemTileDMAOp,
        NextBDOp,
        ObjectFifoAcquireOp,
        ObjectFifoRegisterExternalBuffersOp,
        ObjectFIFOSubviewAccessOp,
        createObjectFifo,
        ObjectFIFOReleaseOp,
        PLIOOp,
        PacketDestOp,
        PacketFlowOp,
        PacketRuleOp,
        PacketRulesOp,
        PacketSourceOp,
        PutCascade,
        PutStream,
        ShimDMAOp,
        ShimMuxOp,
        ShimSwitchBoxOp,
        SwitchboxOp,
        UseLockOp,
        WireOp,
        EndOp,
    ],
    [
        WireBundleAttr,
        ObjectFIFO,
        ObjectFIFOSubview,
    ],
)
