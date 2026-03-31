from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from enum import Enum
from typing import Generic

from typing_extensions import TypeVar

from xdsl.dialects import llvm
from xdsl.dialects.builtin import (
    AnyFloat,
    IntegerType,
    MemRefType,
    Signedness,
    StringAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    attr_def,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_result_def,
    result_def,
)
from xdsl.utils.hints import isa

t_bool: IntegerType = IntegerType(1, Signedness.SIGNLESS)

AnyNumericType = AnyFloat | IntegerType


@irdl_attr_definition
class OperationType(ParametrizedAttribute, TypeAttribute):
    """
    This type represents the MPI_Op type.

    They are used by the reduction MPI functions
    """

    name = "mpi.operation"

    op_str: StringAttr


class MpiOp:
    """
    A collection of MPI_Op types used for
    """

    MPI_MAX = OperationType(StringAttr("MPI_MAX"))
    MPI_MIN = OperationType(StringAttr("MPI_MIN"))
    MPI_SUM = OperationType(StringAttr("MPI_SUM"))
    MPI_PROD = OperationType(StringAttr("MPI_PROD"))
    MPI_LAND = OperationType(StringAttr("MPI_LAND"))
    MPI_BAND = OperationType(StringAttr("MPI_BAND"))
    MPI_LOR = OperationType(StringAttr("MPI_LOR"))
    MPI_BOR = OperationType(StringAttr("MPI_BOR"))
    MPI_LXOR = OperationType(StringAttr("MPI_LXOR"))
    MPI_BXOR = OperationType(StringAttr("MPI_BXOR"))
    MPI_MINLOC = OperationType(StringAttr("MPI_MINLOC"))
    MPI_MAXLOC = OperationType(StringAttr("MPI_MAXLOC"))
    MPI_REPLACE = OperationType(StringAttr("MPI_REPLACE"))
    MPI_NO_OP = OperationType(StringAttr("MPI_NO_OP"))


@irdl_attr_definition
class RequestType(ParametrizedAttribute, TypeAttribute):
    """
    This type represents the MPI_Request type.

    They are used by the asynchronous MPI functions
    """

    name = "mpi.request"


@irdl_attr_definition
class StatusType(ParametrizedAttribute, TypeAttribute):
    """
    This type represents the MPI_Status type.

    It's a struct containing status information for requests.
    """

    name = "mpi.status"


@irdl_attr_definition
class DataType(ParametrizedAttribute, TypeAttribute):
    """
    This type represents MPI_Datatype
    """

    name = "mpi.datatype"


VectorWrappable = RequestType | StatusType | DataType
VectorWrappableConstr = base(RequestType) | base(StatusType) | base(DataType)
_VectorT = TypeVar("_VectorT", bound=VectorWrappable, default=VectorWrappable)


@irdl_attr_definition
class VectorType(ParametrizedAttribute, TypeAttribute, Generic[_VectorT]):
    """
    This type holds multiple MPI types
    """

    name = "mpi.vector"
    wrapped_type: _VectorT


class StatusTypeField(Enum):
    """
    This enum lists all fields in the MPI_Status struct
    """

    MPI_SOURCE = "MPI_SOURCE"
    MPI_TAG = "MPI_TAG"
    MPI_ERROR = "MPI_ERROR"


class MPIBaseOp(IRDLOperation, ABC):
    """
    Base class for MPI Operations
    """

    pass


@irdl_op_definition
class ReduceOp(MPIBaseOp):
    """
    This wraps the MPI_Reduce function (blocking reduction)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Reduce.html).

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

    send_buffer = operand_def(Attribute)
    recv_buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    operationtype = attr_def(OperationType)
    root = operand_def(i32)

    def __init__(
        self,
        send_buffer: SSAValue | Operation,
        recv_buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        operationtype: OperationType,
        root: SSAValue | Operation,
    ):
        return super().__init__(
            operands=[send_buffer, recv_buffer, count, datatype, root],
            attributes={"operationtype": operationtype},
            result_types=[],
        )


@irdl_op_definition
class AllreduceOp(MPIBaseOp):
    """
    This wraps the MPI_Allreduce function (blocking all reduction)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Allreduce.html).

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

    send_buffer = opt_operand_def(Attribute)
    recv_buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    operationtype = attr_def(OperationType)

    def __init__(
        self,
        send_buffer: SSAValue | Operation | None,
        recv_buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        operationtype: OperationType,
    ):
        operands_to_add: Sequence[
            SSAValue | Operation | Sequence[SSAValue | Operation]
        ] = []

        if send_buffer is None:
            operands_to_add = [[], recv_buffer, count, datatype]
        else:
            operands_to_add = [[send_buffer], recv_buffer, count, datatype]

        return super().__init__(
            operands=operands_to_add,
            attributes={"operationtype": operationtype},
            result_types=[],
        )


@irdl_op_definition
class BcastOp(MPIBaseOp):
    """
    This wraps the MPI_Bcast function (blocking broadcast)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Bcast.html).

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

    buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    root = operand_def(i32)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        root: SSAValue | Operation,
    ):
        return super().__init__(
            operands=[buffer, count, datatype, root],
            result_types=[],
        )


@irdl_op_definition
class IsendOp(MPIBaseOp):
    """
    This wraps the MPI_Isend function (nonblocking send)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Isend.html).

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

    name = "mpi.isend"

    buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    dest = operand_def(i32)
    tag = operand_def(i32)
    request = operand_def(RequestType)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        dest: SSAValue | Operation,
        tag: SSAValue | Operation,
        request: SSAValue | Operation,
    ):
        return super().__init__(
            operands=[buffer, count, datatype, dest, tag, request],
            result_types=[],
        )


@irdl_op_definition
class SendOp(MPIBaseOp):
    """
    This wraps the MPI_Send function (blocking send)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Send.html).

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

    name = "mpi.send"

    buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    dest = operand_def(i32)
    tag = operand_def(i32)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        dest: SSAValue | Operation,
        tag: SSAValue | Operation,
    ):
        return super().__init__(
            operands=[buffer, count, datatype, dest, tag], result_types=[]
        )


@irdl_op_definition
class IrecvOp(MPIBaseOp):
    """
    This wraps the MPI_Irecv function (nonblocking receive).

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Irecv.html).

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

    buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    source = operand_def(i32)
    tag = operand_def(i32)
    request = operand_def(RequestType)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        source: SSAValue | Operation,
        tag: SSAValue | Operation,
        request: SSAValue | Operation,
    ):
        return super().__init__(
            operands=[buffer, count, datatype, source, tag, request],
            result_types=[],
        )


@irdl_op_definition
class RecvOp(MPIBaseOp):
    """
    This wraps the MPI_Recv function (blocking receive).

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Recv.html).

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

    buffer = operand_def(Attribute)
    count = operand_def(i32)
    datatype = operand_def(DataType)
    source = operand_def(i32)
    tag = operand_def(i32)

    status = opt_result_def(StatusType)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        count: SSAValue | Operation,
        datatype: SSAValue | Operation,
        source: SSAValue | Operation,
        tag: SSAValue | Operation,
        ignore_status: bool = True,
    ):
        return super().__init__(
            operands=[buffer, count, datatype, source, tag],
            result_types=[[]] if ignore_status else [[StatusType()]],
        )


@irdl_op_definition
class TestOp(MPIBaseOp):
    """
    Class for wrapping the MPI_Test function (test for completion of request)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Test.html).

    ## The MPI_Test Function Docs:

    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)

        - request: Communication request (handle)
        - flag: true if operation completed (logical)
        - status: Status object (Status)
    """

    name = "mpi.test"

    request = operand_def(RequestType)

    flag = result_def(t_bool)
    status = result_def(StatusType)

    def __init__(self, request: Operand):
        return super().__init__(operands=[request], result_types=[t_bool, StatusType()])


@irdl_op_definition
class WaitOp(MPIBaseOp):
    """
    Class for wrapping the MPI_Wait function (blocking wait for request)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Wait.html).

    ## The MPI_Test Function Docs:

    int MPI_Wait(MPI_Request *request, MPI_Status *status)

        - request: Request (handle)
        - status: Status object (Status)
    """

    name = "mpi.wait"

    request = operand_def(RequestType)
    status = opt_result_def(StatusType)

    def __init__(self, request: Operand, ignore_status: bool = True):
        result_types: list[list[Attribute]] = [[StatusType()]]
        if ignore_status:
            result_types = [[]]

        return super().__init__(operands=[request], result_types=result_types)


@irdl_op_definition
class WaitallOp(MPIBaseOp):
    """
    Class for wrapping the MPI_Waitall function (blocking wait for requests)

    See external [documentation](https://www.mpich.org/static/docs/v4.1/www3/MPI_Waitall.html).

    ## The MPI_Test Function Docs:

    int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status *array_of_statuses)

        - count: Number of handles
        - array_of_requests: Request handles
        - array_of_statuses: Status objects
    """

    name = "mpi.waitall"

    requests = operand_def(VectorType[RequestType])
    count = operand_def(i32)
    statuses = opt_result_def(VectorType[StatusType])

    def __init__(self, requests: Operand, count: Operand, ignore_status: bool = True):
        result_types: list[list[Attribute]] = [[VectorType(StatusType())]]
        if ignore_status:
            result_types = [[]]

        return super().__init__(operands=[requests, count], result_types=result_types)


@irdl_op_definition
class GetStatusFieldOp(MPIBaseOp):
    """
    Accessors for the MPI_Status struct

    This allows access to the three integer properties in the struct called
        - MPI_SOURCE
        - MPI_TAG
        - MPI_ERROR

    All fields are of type int.
    """

    name = "mpi.status.get"

    status = operand_def(StatusType)

    field = attr_def(StringAttr)

    result = result_def(i32)

    def __init__(self, status_obj: Operand, field: StatusTypeField):
        return super().__init__(
            operands=[status_obj],
            attributes={"field": StringAttr(field.value)},
            result_types=[i32],
        )


@irdl_op_definition
class CommRankOp(MPIBaseOp):
    """
    Represents the MPI_Comm_size(MPI_Comm comm, int *rank) function call which returns
    the rank of the communicator

    Currently limited to COMM_WORLD
    """

    name = "mpi.comm.rank"

    rank = result_def(i32)

    def __init__(self):
        return super().__init__(result_types=[i32])


@irdl_op_definition
class CommSizeOp(MPIBaseOp):
    """
    Represents the MPI_Comm_size(MPI_Comm comm, int *size) function call which returns
    the size of the communicator

    Currently limited to COMM_WORLD
    """

    name = "mpi.comm.size"

    size = result_def(i32)

    def __init__(self):
        return super().__init__(result_types=[i32])


@irdl_op_definition
class InitOp(MPIBaseOp):
    """
    This represents a bare MPI_Init call with both args being nullptr
    """

    name = "mpi.init"


@irdl_op_definition
class FinalizeOp(MPIBaseOp):
    """
    This represents an MPI_Finalize call with both args being nullptr
    """

    name = "mpi.finalize"


@irdl_op_definition
class UnwrapMemRefOp(MPIBaseOp):
    """
    This Op can be used as a helper to get memrefs into MPI calls.

    It takes any MemRef as input, and returns an llvm.ptr, element count and MPI_Datatype.
    """

    name = "mpi.unwrap_memref"

    ref = operand_def(MemRefType[AnyNumericType])

    ptr = result_def(llvm.LLVMPointerType)
    len = result_def(i32)
    type = result_def(DataType)

    def __init__(self, ref: SSAValue | Operation):
        return super().__init__(
            operands=[ref],
            result_types=[llvm.LLVMPointerType(), i32, DataType()],
        )


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

    dtype = attr_def()

    result = result_def(DataType)

    def __init__(self, dtype: Attribute):
        return super().__init__(result_types=[DataType()], attributes={"dtype": dtype})


@irdl_op_definition
class AllocateTypeOp(MPIBaseOp):
    """
    This op is used to allocate a specific MPI dialect type with a set size,
    returning this in an MPI vector of that type

    This is useful as it means we can, in a self contained manner, store things like
    requests, statuses etc. It accepts the base type that the array will contain, the
    number of elements and an optional 'bindc_name' which contains the name of the
    variable that this is allocating
    """

    name = "mpi.allocate"

    bindc_name = opt_attr_def(StringAttr)
    dtype = attr_def(VectorWrappableConstr)
    count = operand_def(i32)

    result = result_def(VectorType)

    def __init__(
        self,
        dtype: type[VectorWrappable],
        count: SSAValue | Operation,
        bindc_name: StringAttr | None = None,
    ):
        return super().__init__(
            result_types=[VectorType(dtype())],
            attributes={
                "dtype": dtype(),
                "bindc_name": bindc_name,
            },
            operands=[count],
        )


@irdl_op_definition
class VectorGetOp(MPIBaseOp):
    """
    This op will retrieve an element of an MPI vector, it accepts the vector as
    an argument and the element index
    """

    name = "mpi.vector_get"

    vect = operand_def(VectorType)
    element = operand_def(i32)

    result = result_def(VectorWrappableConstr)

    def __init__(self, vect: SSAValue | Operation, element: SSAValue | Operation):
        ssa_val = SSAValue.get(vect)
        assert isa(ssa_val.type, VectorType[VectorWrappable])

        return super().__init__(
            result_types=[ssa_val.type.wrapped_type], operands=[vect, element]
        )


@irdl_op_definition
class NullRequestOp(MPIBaseOp):
    """
    This sets a given request object to the MPI_REQUEST_NULL magic
    value.

    Due to restrictions in the current MPI dialect, we can't return a
    new request object here. That will be fixed soon though!
    """

    name = "mpi.request_null"

    request = operand_def(RequestType)

    def __init__(self, req: SSAValue | Operation):
        return super().__init__(operands=[req])


@irdl_op_definition
class GatherOp(MPIBaseOp):
    """
    This is used to gather data into one big buffer.

    int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root,
                   MPI_Comm comm)

     - sendbuf, sendcount, sendtype: info on the buffer to be sent
     - recvbuf, recvcount, recvtype: info on the gather buffer
     - root: the rank that receives all the data
     - comm: the communicator to use

    Note: recvcount * sizeof(recvtype) == sendcount * sizeof(sendtype) * N
    where N is the number of nodes participating in the gather.

    Note that the data in the recvbuff will not be a nice grid, instead it will contain
    the sent buffers in order of rank.
    """

    name = "mpi.gather"

    sendbuf = operand_def(llvm.LLVMPointerType)
    sendcount = operand_def(i32)
    sendtype = operand_def(DataType)

    recvbuf = operand_def(llvm.LLVMPointerType)
    recvcount = operand_def(i32)
    recvtype = operand_def(DataType)

    root = operand_def(i32)

    def __init__(
        self,
        sendbuf: SSAValue | Operation,
        sendcount: SSAValue | Operation,
        sendtype: SSAValue | Operation,
        recvbuf: SSAValue | Operation,
        recvcount: SSAValue | Operation,
        recvtype: SSAValue | Operation,
        root: SSAValue | Operation,
    ):
        super().__init__(
            operands=[
                sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                root,
            ]
        )


MPI = Dialect(
    "mpi",
    [
        CommSizeOp,
        IsendOp,
        IrecvOp,
        TestOp,
        RecvOp,
        SendOp,
        ReduceOp,
        AllreduceOp,
        BcastOp,
        WaitOp,
        WaitallOp,
        GetStatusFieldOp,
        InitOp,
        FinalizeOp,
        CommRankOp,
        UnwrapMemRefOp,
        GetDtypeOp,
        AllocateTypeOp,
        VectorGetOp,
        NullRequestOp,
        GatherOp,
    ],
    [
        OperationType,
        RequestType,
        StatusType,
        DataType,
        VectorType,
    ],
)
