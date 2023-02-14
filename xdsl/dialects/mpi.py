from dialects.arith import Constant
from xdsl.ir import Operation, OpResult, ParametrizedAttribute, Region
from xdsl.irdl import Operation, OpResult, Operand, Annotated, AnyAttr, Attribute, irdl_op_definition, \
    irdl_attr_definition, OptOpAttr
from xdsl.dialects.builtin import IntegerType, i64, Signedness, IntegerAttr, f64, AnyFloatAttr, AnyIntegerAttr
from xdsl.printer import Printer
from xdsl.dialects.memref import MemRefType, Alloc
from abc import ABC

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


if __name__ == '__main__':
    printer = Printer(target=Printer.Target.MLIR)

    # yapf: ignore
    reg = Region.from_operation_list([
        memref := Alloc.get(f64, 32, [100, 14, 14]),
        dest := Constant.from_int_and_width(1,t_int),
        req := ISend.get(memref, dest, 1),
        res := IRecv.get(dest, memref.results[0].typ, 1),
        test_res := Test.get(res.results[1]),
        flag := StatusGetFlag.get(test_res),
        code := StatusGetStatus.get(test_res),
        code2 := Wait.get(res.results[1])
    ])  # yapf: disable

    printer.print_region(reg)
"""
// Example isend
// %in is the input memref
// %dest is a destination rank (si32)
%request = "mpi.isend"(%in, %dest) {"tag" = 1} : (!memref<3x2x2xi64>, !si32) -> (!mpi.request) 


// example irecv
// %source is the source rank (si32)
%data, %request = "mpi.irecv"(%source) {"tag" = 1} : (!si32) -> (!memref<3x2x2xi64>, !mpi.request)

// example test
// %request is an !mpi.request
%status_obj = "mpi.test"(%request) : (!mpi.request) -> !mpi.status
%flag = "mpi.get_status_flag"(%status_obj) : (!mpi.status) -> i1
%status = "mpi.get_status_code"(%status_obj) : (!mpi.status) -> si32

// example wait
// %request is an !mpi.request
%status = "mpi.wait"(%request) : (!mpi.request) -> si32
"""
