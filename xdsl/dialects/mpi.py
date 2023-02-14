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


class MPIBaseOp(Operation, ABC):
    pass


@irdl_attr_definition
class RequestType(ParametrizedAttribute):
    name = 'mpi.request'


@irdl_attr_definition
class StatusType(ParametrizedAttribute):
    name = 'mpi.status'


@irdl_op_definition
class ISend(MPIBaseOp):
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
class IRecv(MPIBaseOp):
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
class Test(MPIBaseOp):
    name = "mpi.test"

    request: Annotated[Operand, RequestType()]
    status: Annotated[OpResult, StatusType()]

    @classmethod
    def get(cls, request: Operand):
        return cls.build(operands=[request], result_types=[StatusType()])


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
