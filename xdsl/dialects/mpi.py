from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import cast

from xdsl.dialects import llvm
from xdsl.dialects.builtin import (IntegerType, Signedness, IntegerAttr,
                                   StringAttr, AnyFloat, i32)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, Attribute, SSAValue, OpResult, ParametrizedAttribute, Dialect, MLIRType
from xdsl.irdl import (Operand, Annotated, irdl_op_definition, AnyAttr,
                       irdl_attr_definition, OpAttr, OptOpResult)

t_bool: IntegerType = IntegerType(1, Signedness.SIGNLESS)

AnyNumericType = AnyFloat | IntegerType


@irdl_attr_definition
class RequestType(ParametrizedAttribute, MLIRType):
    """
    This type represents the MPI_Request type.

    They are used by the asynchronous MPI functions
    """
    name = 'mpi.request'


@irdl_attr_definition
class StatusType(ParametrizedAttribute, MLIRType):
    """
    This type represents the MPI_Status type.

    It's a struct containing status information for requests.
    """
    name = 'mpi.status'


@irdl_attr_definition
class DataType(ParametrizedAttribute, MLIRType):
    """
    This type represents MPI_Datatype
    """
    name = 'mpi.datatype'


class StatusTypeField(Enum):
    """
    This enum lists all fields in the MPI_Status struct
    """
    MPI_SOURCE = 'MPI_SOURCE'
    MPI_TAG = 'MPI_TAG'
    MPI_ERROR = 'MPI_ERROR'


class MPIBaseOp(Operation, ABC):
    """
    Base class for MPI Operations
    """
    pass


def _build_attr_dict_with_optional_tag(
        tag: int | None = None) -> dict[str, Attribute]:
    """
    Helper function for building attribute dicts that have an optional `tag` entry
    """

    return {} if tag is None else {'tag': IntegerAttr.from_params(tag, i32)}


@irdl_op_definition
class ISend(MPIBaseOp):
    """
    This wraps the MPI_Isend function (nonblocking send)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Isend.html

    ## The MPI_Isend Function Docs:

    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
         int tag, MPI_Comm comm, MPI_Request *request)

        - buf: Initial address of send buffer (choice).
        - count: Number of elements in send buffer (integer).
        - datatype: Datatype of each send buffer element (handle).
        - dest: Rank of destination (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).

    ## Our Abstraction:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = 'mpi.isend'

    buffer: Annotated[Operand, AnyAttr()]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    dest: Annotated[Operand, i32]
    tag: Annotated[Operand, i32]

    request: Annotated[OpResult, RequestType]

    @classmethod
    def get(cls, buff: SSAValue | Operation, dest: SSAValue | Operation,
            tag: int | None):
        return cls.build(operands=[buff, dest],
                         attributes=_build_attr_dict_with_optional_tag(tag),
                         result_types=[RequestType()])


@irdl_op_definition
class Send(MPIBaseOp):
    """
    This wraps the MPI_Send function (blocking send)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Send.html

    ## The MPI_Send Function Docs:

    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)

        - buf: Initial address of send buffer (choice).
        - count: Number of elements in send buffer (non-negative integer).
        - datatype: Datatype of each send buffer element (handle).
        - dest: Rank of destination (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).

    ## Our Abstraction:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = 'mpi.send'

    buffer: Annotated[Operand, AnyAttr()]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    dest: Annotated[Operand, i32]
    tag: Annotated[Operand, i32]

    @classmethod
    def get(cls, buffer: SSAValue | Operation, count: SSAValue | Operation,
            datatype: SSAValue | Operation, dest: SSAValue | Operation,
            tag: SSAValue | Operation) -> Send:
        return cls.build(operands=[buffer, count, datatype, dest, tag],
                         result_types=[])


@irdl_op_definition
class IRecv(MPIBaseOp):
    """
    This wraps the MPI_Irecv function (nonblocking receive).
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Irecv.html

    ## The MPI_Irecv Function Docs:

    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
            int source, int tag, MPI_Comm comm, MPI_Request *request)

        - buf: Initial address of receive buffer (choice).
        - count: Number of elements in receive buffer (integer).
        - datatype: Datatype of each receive buffer element (handle).
        - source: Rank of source (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).
        - request: Communication request (handle).

    ## Our Abstractions:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.irecv"

    buffer: Annotated[Operand, AnyAttr()]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    source: Annotated[Operand, i32]
    tag: Annotated[Operand, i32]

    request: Annotated[OpResult, RequestType]

    @classmethod
    def get(cls,
            source: SSAValue | Operation,
            buffer: SSAValue | Operation,
            tag: int | None = None):
        return cls.build(operands=[source, buffer],
                         attributes=_build_attr_dict_with_optional_tag(tag),
                         result_types=[RequestType()])


@irdl_op_definition
class Recv(MPIBaseOp):
    """
    This wraps the MPI_Recv function (blocking receive).
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Recv.html

    ## The MPI_Recv Function Docs:

    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)

        - buf: Initial address of receive buffer (choice).
        - count: Number of elements in receive buffer (integer).
        - datatype: Datatype of each receive buffer element (handle).
        - source: Rank of source (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).
        - status: status object (Status).

    ## Our Abstractions:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.recv"

    buffer: Annotated[Operand, AnyAttr()]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    source: Annotated[Operand, i32]
    tag: Annotated[Operand, i32]

    status: Annotated[OptOpResult, StatusType]

    @classmethod
    def get(cls,
            buffer: SSAValue | Operation,
            count: SSAValue | Operation,
            datatype: SSAValue | Operation,
            source: SSAValue | Operation,
            tag: SSAValue | Operation,
            ignore_status: bool = True):
        return cls.build(
            operands=[buffer, count, datatype, source, tag],
            result_types=[[]] if ignore_status else [[StatusType()]])


@irdl_op_definition
class Test(MPIBaseOp):
    """
    Class for wrapping the MPI_Test function (test for completion of request)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Test.html

    ## The MPI_Test Function Docs:

    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)

        - request: Communication request (handle)
        - flag: true if operation completed (logical)
        - status: Status object (Status)
    """

    name = "mpi.test"

    request: Annotated[Operand, RequestType]

    flag: Annotated[OpResult, t_bool]
    status: Annotated[OpResult, StatusType]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request],
                         result_types=[t_bool, StatusType()])


@irdl_op_definition
class Wait(MPIBaseOp):
    """
    Class for wrapping the MPI_Wait function (blocking wait for request)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Wait.html

    ## The MPI_Test Function Docs:

    int MPI_Wait(MPI_Request *request, MPI_Status *status)

        - request: Request (handle)
        - status: Status object (Status)
    """

    name = "mpi.wait"

    request: Annotated[Operand, RequestType]
    status: Annotated[OptOpResult, StatusType]

    @classmethod
    def get(cls, request: Operand, ignore_status: bool = True):
        result_types: list[list[Attribute]] = [[StatusType()]]
        if ignore_status:
            result_types = [[]]

        return cls.build(operands=[request], result_types=result_types)


@irdl_op_definition
class GetStatusField(MPIBaseOp):
    """
    Accessors for the MPI_Status struct

    This allows access to the three integer properties in the struct called
        - MPI_SOURCE
        - MPI_TAG
        - MPI_ERROR

    All fields are of type int.
    """
    name = "mpi.status.get"

    status: Annotated[Operand, StatusType]

    field: OpAttr[StringAttr]

    result: Annotated[OpResult, i32]

    @classmethod
    def get(cls, status_obj: Operand, field: StatusTypeField):
        return cls.build(operands=[status_obj],
                         attributes={'field': StringAttr(field.value)},
                         result_types=[i32])


@irdl_op_definition
class CommRank(MPIBaseOp):
    """
    Represents the MPI_Comm_size(MPI_Comm comm, int *rank) function call which returns the rank of the communicator

    Currently limited to COMM_WORLD
    """
    name = "mpi.comm.rank"

    rank: Annotated[OpResult, i32]

    @classmethod
    def get(cls):
        return cls.build(result_types=[i32])


@irdl_op_definition
class CommSize(MPIBaseOp):
    """
    Represents the MPI_Comm_size(MPI_Comm comm, int *size) function call which returns the size of the communicator

    Currently limited to COMM_WORLD
    """
    name = "mpi.comm.size"

    size: Annotated[OpResult, i32]

    @classmethod
    def get(cls):
        return cls.build(result_types=[i32])


@irdl_op_definition
class Init(MPIBaseOp):
    """
    This represents a bare MPI_Init call with both args being nullptr
    """
    name = "mpi.init"


@irdl_op_definition
class Finalize(MPIBaseOp):
    """
    This represents an MPI_Finalize call with both args being nullptr
    """
    name = "mpi.finalize"


@irdl_op_definition
class UnwrapMemrefOp(MPIBaseOp):
    """
    This Op can be used as a helper to get memrefs into MPI calls.

    It takes any MemRef as input, and returns an llvm.ptr, element count and MPI_Datatype.
    """
    name = "mpi.unwrap_memref"

    ref: Annotated[Operand, MemRefType[AnyNumericType]]

    ptr: Annotated[OpResult, llvm.LLVMPointerType]
    len: Annotated[OpResult, i32]
    typ: Annotated[OpResult, DataType]

    @staticmethod
    def get(ref: SSAValue | Operation):
        ssa_val = SSAValue.get(ref)
        assert isinstance(ssa_val.typ, MemRefType)
        elem_typ = cast(MemRefType[AnyNumericType], ssa_val.typ).element_type

        return UnwrapMemrefOp.build(operands=[ref],
                                    result_types=[
                                        llvm.LLVMPointerType.typed(elem_typ),
                                        i32,
                                        DataType()
                                    ])


@irdl_op_definition
class GetDtypeOp(MPIBaseOp):
    """
    This op is used to convert MLIR types to MPI_Datatype constants.

    So, e.g. if you want to get the `MPI_Datatype` for an `i32` you can use

        %dtype = "mpi.get_dtype"() {"dtype" = i32} : () -> mpi.datatype

    to get the magic constant. See `_MPIToLLVMRewriteBase._translate_to_mpi_type`
    docstring for more detail on which types are supported.
    """
    name = "mpi.get_dtype"

    dtype: OpAttr[Attribute]

    result: Annotated[OpResult, DataType]

    @staticmethod
    def get(typ: Attribute):
        return GetDtypeOp.build(result_types=[DataType()],
                                attributes={'dtype': typ})


MPI = Dialect([
    ISend,
    IRecv,
    Test,
    Recv,
    Send,
    GetStatusField,
    Init,
    Finalize,
    CommRank,
    UnwrapMemrefOp,
    GetDtypeOp,
], [
    RequestType,
    StatusType,
    DataType,
])
