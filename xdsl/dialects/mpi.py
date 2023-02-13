from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, List, TYPE_CHECKING, Union

from xdsl.dialects.builtin import IntegerAttr, Attribute
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Data, OpResult, SSAValue, Dialect
from xdsl.ir import ParametrizedAttribute

from xdsl.irdl import (OpAttr, irdl_op_definition, Block, Region, Operation, Operand, OpResult,
                       builder, irdl_attr_definition, AnyAttr)

if TYPE_CHECKING:
    from xdsl.parser import BaseParser
    from utils.exceptions import ParseError
    from xdsl.printer import Printer


@irdl_op_definition
class MPI_ISend(Operation):
    name: str = "mpi.MPI_ISend"

    buffer: Annotated[Operand, MemRefType]
    count: Annotated[Operand, IntegerAttr]
    recipient: Annotated[Operand, IntegerAttr]
    tag: Annotated[Operand, IntegerAttr]
    communicator: Annotated[Operand, IntegerAttr]
    request: Annotated[Operand, IntegerAttr]


    @staticmethod
    def from_callable(buffer: Union[Operation, SSAValue],
                      count: Union[Operation, SSAValue],
                      recipient: Union[Operation, SSAValue],
                      tag: Union[Operation, SSAValue],
                      communicator: Union[Operation, SSAValue],
                      request: Union[Operation, SSAValue]) -> MPI_ISend:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": name,
            "function_type": type_attr,
            "sym_visibility": "private"
        }
        op = MPI_ISend.build(attributes=attributes,
                             regions=[Region.from_block_list(
                                     [Block.from_callable(input_types, func)])])
        return op


@irdl_attr_definition
class MPIIntAttr(Data[int]):
    name = "mpi_int"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> int:
        data = parser.parse_int_literal()
        return data

    @staticmethod
    def print_parameter(data: int, printer: Printer) -> None:
        printer.print_string(f'{data}')

    @staticmethod
    @builder
    def from_int(data: int) -> MPIIntAttr:
        return MPIIntAttr(data)


@irdl_attr_definition
class MPIRequest(ParametrizedAttribute):
    name = "MPI_Request"


@irdl_attr_definition
class MPIStatus(ParametrizedAttribute):
    name = "MPI_Status"


@irdl_op_definition
class MPIRequestOp(Operation):
    # Name of the type. This is used for printing and parsing.
    name: str = "mpi.MPIRequestOp"
    request: Annotated[Operand, AnyAttr()]
    # output: Annotated[OpResult, Attribute]

    @staticmethod
    def from_attr(request: AnyAttr) -> MPIRequestOp:
        return MPIRequestOp.create(operands=[],
                                   result_types=[request])


@irdl_op_definition
class MPI_Wait(Operation):
    name: str = "mpi.MPI_Wait"

    request: Annotated[Operand, AnyAttr()]
    status: Annotated[OpResult, MPIStatus]

    @staticmethod
    def from_callable(request: Operation) -> MPI_Wait:
        op1 = SSAValue.get(request)
        return MPI_Wait.build(operands=[op1], result_types=[MPIStatus()])


MPI = Dialect(
    [MPI_Wait, MPI_ISend, MPIRequestOp],
    [
        MPIRequest,
        MPIStatus
    ])
