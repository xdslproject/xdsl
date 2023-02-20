from abc import ABC
from enum import Enum

from xdsl.ir import Attribute, OpResult, ParametrizedAttribute, Dialect, Operation
from xdsl.irdl import (Operand, Annotated, irdl_op_definition,
                       irdl_attr_definition, OptOpAttr, OpAttr)
from xdsl.dialects.builtin import (IntegerType, Signedness, IntegerAttr,
                                   AnyFloatAttr, AnyIntegerAttr, StringAttr)
from xdsl.dialects.memref import MemRefType, Alloc

t_int: IntegerType = IntegerType.from_width(32, Signedness.SIGNLESS)
t_bool: IntegerType = IntegerType.from_width(1, Signedness.SIGNLESS)

AnyNumericAttr = AnyFloatAttr | AnyIntegerAttr


@irdl_attr_definition
class RequestType(ParametrizedAttribute):
    """
    This type represents the MPI_Request type.

    They are used by the asynchronous MPI functions
    """
    name = 'mpi.request'


@irdl_attr_definition
class StatusType(ParametrizedAttribute):
    """
    This type represents the MPI_Status type.

    It's a struct containing status information for requests.
    """
    name = 'mpi.status'


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

    return {} if tag is None else {'tag': IntegerAttr.from_params(tag, t_int)}


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

        - We Summarize buf, count and datatype by using memrefs
        - We assume that tag is compile-time constant
        - We omit the possibility of using multiple communicators
    """

    name = 'mpi.isend'

    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]
    dest: Annotated[Operand, t_int]

    tag: OptOpAttr[AnyIntegerAttr]

    request: Annotated[OpResult, RequestType()]

    @classmethod
    def get(cls, buff: Operand, dest: Operand, tag: int | None):
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

        - We Summarize buf, count and datatype by using memrefs
        - We assume that tag is compile-time constant
        - We omit the possibility of using multiple communicators
    """

    name = 'mpi.send'

    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]
    dest: Annotated[Operand, t_int]

    tag: OptOpAttr[AnyIntegerAttr]

    @classmethod
    def get(cls, buff: Operand, dest: Operand, tag: int | None):
        return cls.build(operands=[buff, dest],
                         attributes=_build_attr_dict_with_optional_tag(tag),
                         result_types=[RequestType()])


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

        - We bundle buf, count and datatype into the type definition and use `memref`
        - We assume this type information is compile-time known
        - We assume tag is compile-time known
        - We omit the possibility of using multiple communicators
    """

    name = "mpi.irecv"

    source: Annotated[Operand, t_int]

    tag: OptOpAttr[AnyIntegerAttr]

    buffer: Annotated[OpResult, MemRefType[AnyNumericAttr]]
    request: Annotated[OpResult, RequestType()]

    @classmethod
    def get(cls,
            source: Operand,
            dtype: MemRefType[AnyNumericAttr],
            tag: int | None = None):
        return cls.build(operands=[source],
                         attributes=_build_attr_dict_with_optional_tag(tag),
                         result_types=[dtype, RequestType()])


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

        - We bundle buf, count and datatype into the type definition and use `memref`
        - We assume this type information is compile-time known
        - We assume tag is compile-time known
        - We omit the possibility of using multiple communicators
    """

    name = "mpi.recv"

    source: Annotated[Operand, t_int]

    tag: OptOpAttr[AnyIntegerAttr]

    buffer: Annotated[OpResult, MemRefType[AnyNumericAttr]]
    status: Annotated[OpResult, StatusType]

    @classmethod
    def get(cls,
            source: Operand,
            dtype: MemRefType[AnyNumericAttr],
            tag: int | None = None):
        return cls.build(operands=[source],
                         attributes=_build_attr_dict_with_optional_tag(tag),
                         result_types=[dtype, StatusType()])


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

    request: Annotated[Operand, RequestType()]
    status: Annotated[OpResult, t_int]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request], result_types=[t_int])


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

    result: Annotated[OpResult, t_int]

    @classmethod
    def get(cls, status_obj: Operand, field: StatusTypeField):
        return cls.build(operands=[status_obj],
                         attributes={'field': StringAttr(field.value)},
                         result_types=[t_int])


@irdl_op_definition
class Init(MPIBaseOp):
    name = "mpi.init"


@irdl_op_definition
class Finalize(MPIBaseOp):
    name = "mpi.finalize"


MPI = Dialect([
    MPIBaseOp, Alloc, ISend, IRecv, Test, Recv, Send, GetStatusField, Init,
    Finalize
], [
    RequestType,
])
