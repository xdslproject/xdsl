from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import cast

from xdsl.dialects import llvm
from xdsl.dialects.builtin import (IntegerType, Signedness, StringAttr,
                                   AnyFloat, i32)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import (Operation, Attribute, SSAValue, OpResult,
                     ParametrizedAttribute, Dialect, MLIRType)
from xdsl.irdl import (Operand, Annotated, irdl_op_definition,
                       irdl_attr_definition, OpAttr, OptOpResult, ParameterDef,
                       AnyOf, OptOperand)

t_bool: IntegerType = IntegerType(1, Signedness.SIGNLESS)

AnyNumericType = AnyFloat | IntegerType


@irdl_attr_definition
class OperationType(ParametrizedAttribute, MLIRType):
    """
    This type represents the MPI_Op type.

    They are used by the reduction MPI functions
    """
    name = 'mpi.operation'

    op_str: ParameterDef[StringAttr]


MPI_MAX = OperationType([StringAttr("MPI_MAX")])
MPI_MIN = OperationType([StringAttr("MPI_MIN")])
MPI_SUM = OperationType([StringAttr("MPI_SUM")])
MPI_PROD = OperationType([StringAttr("MPI_PROD")])
MPI_LAND = OperationType([StringAttr("MPI_LAND")])
MPI_BAND = OperationType([StringAttr("MPI_BAND")])
MPI_LOR = OperationType([StringAttr("MPI_LOR")])
MPI_BOR = OperationType([StringAttr("MPI_BOR")])
MPI_LXOR = OperationType([StringAttr("MPI_LXOR")])
MPI_BXOR = OperationType([StringAttr("MPI_BXOR")])
MPI_MINLOC = OperationType([StringAttr("MPI_MINLOC")])
MPI_MAXLOC = OperationType([StringAttr("MPI_MAXLOC")])
MPI_REPLACE = OperationType([StringAttr("MPI_REPLACE")])
MPI_NO_OP = OperationType([StringAttr("MPI_NO_OP")])


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


@irdl_attr_definition
class VectorType(ParametrizedAttribute, MLIRType):
    """
    This type holds multiple MPI types
    """
    name = 'mpi.vector'
    typ: ParameterDef[RequestType | StatusType | DataType]


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


@irdl_op_definition
class Reduce(MPIBaseOp):
    """
    This wraps the MPI_Reduce function (blocking reduction)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Reduce.html

    ## The MPI_Reduce Function Docs:

    int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)

        sendbuf: address of send buffer (choice)
        recvbuf: address of receive buffer (choice)
        count: number of elements in send buffer (non-negative integer)
        datatype: data type of elements of send buffer (handle)
        op: reduce operation (handle)
        root: rank of root process (integer)
        comm: communicator (handle)

    ## Our Abstraction:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.reduce"

    send_buffer: Annotated[Operand, Attribute]
    recv_buffer: Annotated[Operand, Attribute]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    operationtype: OpAttr[OperationType]
    root: Annotated[Operand, i32]

    @classmethod
    def get(
        cls,
        send_buffer: SSAValue | Operation,
        recv_buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        operationtype: OperationType,
        root: SSAValue | Operation,
    ):
        return cls.build(
            operands=[send_buffer, recv_buffer, count, datatype, root],
            attributes={"operationtype": operationtype},
            result_types=[],
        )


@irdl_op_definition
class Allreduce(MPIBaseOp):
    """
    This wraps the MPI_Allreduce function (blocking all reduction)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Allreduce.html

    ## The MPI_Allreduce Function Docs:

    int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)

        sendbuf: address of send buffer (choice)
        recvbuf: address of receive buffer (choice)
        count: number of elements in send buffer (non-negative integer)
        datatype: data type of elements of send buffer (handle)
        op: reduce operation (handle)
        comm: communicator (handle)

    ## Our Abstraction:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.allreduce"

    recv_buffer: Annotated[Operand, Attribute]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    send_buffer: Annotated[OptOperand, Attribute]
    operationtype: OpAttr[OperationType]

    @classmethod
    def get(
        cls,
        send_buffer: SSAValue | Operation | None,
        recv_buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        operationtype: OperationType,
    ):

        if send_buffer is None:
            operands_to_add = [recv_buffer, count, datatype, []]
        else:
            operands_to_add = [recv_buffer, count, datatype, [send_buffer]]

        return cls.build(
            operands=operands_to_add,
            attributes={"operationtype": operationtype},
            result_types=[],
        )


@irdl_op_definition
class Bcast(MPIBaseOp):
    """
    This wraps the MPI_Bcast function (blocking broadcast)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Bcast.html

    ## The MPI_Bcast Function Docs:

    int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm)

        buffer: starting address of buffer (choice)
        count: number of elements in send buffer (non-negative integer)
        datatype: data type of elements of send buffer (handle)
        root: rank of broadcast root (integer)
        comm: communicator (handle)

    ## Our Abstraction:

        - We omit the possibility of using multiple communicators, defaulting
          to MPI_COMM_WORLD
    """

    name = "mpi.bcast"

    buffer: Annotated[Operand, Attribute]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    root: Annotated[Operand, i32]

    @classmethod
    def get(
        cls,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        root: SSAValue | Operation,
    ):
        return cls.build(
            operands=[buffer, count, datatype, root],
            result_types=[],
        )


@irdl_op_definition
class Isend(MPIBaseOp):
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

    buffer: Annotated[Operand, Attribute]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    dest: Annotated[Operand, i32]
    tag: Annotated[Operand, i32]
    request: Annotated[Operand, RequestType]

    @classmethod
    def get(
        cls,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        dest: SSAValue | Operation,
        tag: SSAValue | Operation,
        request: SSAValue | Operation,
    ):
        return cls.build(
            operands=[buffer, count, datatype, dest, tag, request],
            result_types=[],
        )


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

    buffer: Annotated[Operand, Attribute]
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
class Irecv(MPIBaseOp):
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

    buffer: Annotated[Operand, Attribute]
    count: Annotated[Operand, i32]
    datatype: Annotated[Operand, DataType]
    source: Annotated[Operand, i32]
    tag: Annotated[Operand, i32]
    request: Annotated[Operand, RequestType]

    @classmethod
    def get(
        cls,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        source: SSAValue | Operation,
        tag: SSAValue | Operation,
        request: SSAValue | Operation,
    ):
        return cls.build(
            operands=[buffer, count, datatype, source, tag, request],
            result_types=[],
        )


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

    buffer: Annotated[Operand, Attribute]
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
class WaitAll(MPIBaseOp):
    """
    Class for wrapping the MPI_Waitall function (blocking wait for requests)
    https://www.mpich.org/static/docs/v4.1/www3/MPI_Waitall.html

    ## The MPI_Test Function Docs:

    int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status *array_of_statuses)

        - count: Number of handles
        - array_of_requests: Request handles
        - array_of_statuses: Status objects
    """

    name = "mpi.waitall"

    requests: Annotated[Operand, VectorType([RequestType()])]
    count: Annotated[Operand, i32]
    statuses: Annotated[OptOpResult, VectorType([StatusType()])]

    @classmethod
    def get(cls,
            requests: Operand,
            count: Operand,
            ignore_status: bool = True):
        result_types: list[list[Attribute]] = [[VectorType([StatusType()])]]
        if ignore_status:
            result_types = [[]]

        return cls.build(operands=[requests, count], result_types=result_types)


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
    def get(ref: SSAValue | Operation) -> UnwrapMemrefOp:
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


@irdl_op_definition
class AllocateTypeOp(MPIBaseOp):
    """
    This op is used to allocate a specific MPI dialect type with a set size, returning this
    in an MPI vector of that type

    This is useful as it means we can, in a self contained manner, store things like
    requests, statuses etc. It accepts the base type that the array will contain, the
    number of elements and an optional bindc_name which contains the name of the
    variable that this is allocating
    """
    name = "mpi.allocate"

    bindc_name: OpAttr[StringAttr]
    dtype: OpAttr[Attribute]
    count: Annotated[Operand, i32]

    result: Annotated[OpResult, VectorType]

    @staticmethod
    def get(bindc_name: StringAttr, dtype: Attribute,
            count: SSAValue | Operation) -> AllocateTypeOp:
        return AllocateTypeOp.build(result_types=[VectorType([dtype])],
                                    attributes={
                                        'bindc_name': bindc_name,
                                        'dtype': dtype
                                    },
                                    operands=[count])


@irdl_op_definition
class VectorGetOp(MPIBaseOp):
    """
    This op will retrieve an element of an MPI vector, it accepts the vector as
    an argument and the element index
    """
    name = "mpi.vector_get"

    vect: Annotated[Operand, VectorType]
    element: Annotated[Operand, i32]

    result: Annotated[OpResult, AnyOf([RequestType, StatusType, DataType])]

    @staticmethod
    def get(vect: VectorType, element: SSAValue | Operation) -> AllocateTypeOp:
        ssa_val = SSAValue.get(vect)
        return VectorGetOp.build(result_types=[ssa_val.typ.typ],
                                 operands=[vect, element])


MPI = Dialect([
    Isend,
    Irecv,
    Test,
    Recv,
    Send,
    Reduce,
    Allreduce,
    Bcast,
    Wait,
    GetStatusField,
    Init,
    Finalize,
    CommRank,
    UnwrapMemrefOp,
    GetDtypeOp,
    AllocateTypeOp,
    VectorGetOp,
], [
    OperationType,
    RequestType,
    StatusType,
    DataType,
])
