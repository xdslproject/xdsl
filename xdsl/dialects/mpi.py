from abc import ABC

from xdsl.ir import OpResult, ParametrizedAttribute, Dialect, Operation
from xdsl.irdl import (Operand, Annotated, irdl_op_definition,
                       irdl_attr_definition, OptOpAttr)
from xdsl.dialects.builtin import (IntegerType, Signedness, IntegerAttr,
                                   AnyFloatAttr, AnyIntegerAttr)
from xdsl.dialects.memref import MemRefType, Alloc

t_uint32: IntegerType = IntegerType.from_width(32, Signedness.UNSIGNED)
t_int: IntegerType = IntegerType.from_width(32, Signedness.SIGNED)
t_bool: IntegerAttr = IntegerType.from_width(1, Signedness.SIGNLESS)

AnyNumericAttr = AnyFloatAttr | AnyIntegerAttr


@irdl_attr_definition
class RequestType(ParametrizedAttribute):
    # Type defined for MPI_Request
    name = 'mpi.request'


@irdl_attr_definition
class StatusType(ParametrizedAttribute):
    # Type defined for MPI_Status
    name = 'mpi.status'


class MPIBaseOp(Operation, ABC):
    # Base class for MPI Operations
    pass


@irdl_op_definition
class ISend(MPIBaseOp):
    """
    This wraps the MPI_Isend function

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

    """
    name = 'mpi.isend'

    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]
    dest: Annotated[Operand, t_int]

    tag: OptOpAttr[IntegerAttr]

    request: Annotated[OpResult, RequestType()]

    @classmethod
    def get(cls, buff: Operand, dest: Operand, tag: int | None):
        attrs = {}

        if tag is not None:
            attrs['tag'] = IntegerAttr.from_params(tag, t_int)

        return cls.build(operands=[buff, dest],
                         attributes=attrs,
                         result_types=[RequestType()])


@irdl_op_definition
class Send(MPIBaseOp):
    # Class for an MPI_Send operation (blocking send)
    # https://www.mpich.org/static/docs/v3.1/www3/MPI_Send.html
    name = 'mpi.send'

    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]
    dest: Annotated[Operand, t_int]

    tag: OptOpAttr[IntegerAttr]

    @classmethod
    def get(cls, buff: Operand, dest: Operand, tag: int | None):
        attrs = {}

        if tag is not None:
            attrs['tag'] = IntegerAttr.from_params(tag, t_int)

        return cls.build(operands=[buff, dest],
                         attributes=attrs,
                         result_types=[RequestType()])


@irdl_op_definition
class IRecv(MPIBaseOp):
    """
    This wraps the MPI_Irecv function.

    ## The MPI_Irecv Function Docs:

    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
            int source, int tag, MPI_Comm comm, MPI_Request *request)

        - buf: Initial address of receive buffer (choice).
        - count: Number of elements in receive buffer (integer).
        - datatype: Datatype of each receive buffer element (handle).
        - source: Rank of source (integer).
        - tag: Message tag (integer).
        - comm: Communicator (handle).

    ## Our Abstractions:

        - We bundle buf, count and datatype into the type definition and use `memref`
        - We assume this type information is compile-time known
        - We assume tag is compile-time known

    """

    name = "mpi.irecv"

    source: Annotated[Operand, t_int]

    tag: OptOpAttr[IntegerAttr]

    buffer: Annotated[OpResult, MemRefType[AnyNumericAttr]]
    request: Annotated[OpResult, RequestType()]

    @classmethod
    def get(cls,
            source: Operand,
            dtype: MemRefType[AnyNumericAttr],
            tag: int | None = None):
        attrs = {}

        if tag is not None:
            attrs['tag'] = IntegerAttr.from_params(tag, t_int)

        return cls.build(operands=[source],
                         attributes=attrs,
                         result_types=[dtype, RequestType()])


@irdl_op_definition
class Recv(MPIBaseOp):
    # Class for an MPI_Recv operation (blocking receive for a message)
    # https://www.mpich.org/static/docs/v3.3/www3/MPI_Recv.html
    name = "mpi.irecv"

    source: Annotated[Operand, t_int]

    tag: OptOpAttr[IntegerAttr]

    buffer: Annotated[OpResult, MemRefType[AnyNumericAttr]]
    status: Annotated[OpResult, StatusType()]

    @classmethod
    def get(cls,
            source: Operand,
            dtype: MemRefType[AnyNumericAttr],
            tag: int | None = None):
        attrs = {}

        if tag is not None:
            attrs['tag'] = IntegerAttr.from_params(tag, t_int)

        return cls.build(operands=[source],
                         attributes=attrs,
                         result_types=[dtype, StatusType()])


@irdl_op_definition
class Test(MPIBaseOp):
    # Class for an MPI_Test operation (Tests for the completion of a request
    # https://www.mpich.org/static/docs/v3.2/www3/MPI_Test.html
    name = "mpi.test"

    flag: Annotated[OpResult, t_bool]
    status: Annotated[OpResult, StatusType()]
    request: Annotated[Operand, RequestType()]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request], result_types=[t_bool, StatusType()])


@irdl_op_definition
class StatusGetFlag(MPIBaseOp):
    name = "mpi.status_get_flag"

    request: Annotated[Operand, StatusType()]
    status: Annotated[OpResult, t_bool]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request], result_types=[t_bool])


@irdl_op_definition
class StatusGetStatus(MPIBaseOp):
    name = "mpi.status_get_status"

    request: Annotated[Operand, StatusType()]
    status: Annotated[OpResult, t_int]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request], result_types=[t_int])


@irdl_op_definition
class Wait(MPIBaseOp):
    name = "mpi.wait"

    request: Annotated[Operand, RequestType()]
    status: Annotated[OpResult, t_int]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request], result_types=[t_int])


MPI = Dialect(
    [MPIBaseOp, Alloc, ISend, IRecv, Test, StatusGetFlag, StatusGetStatus],
    [RequestType, StatusType])
