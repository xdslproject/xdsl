from dialects.arith import Constant
from xdsl.ir import Operation, OpResult, ParametrizedAttribute, Region
from xdsl.irdl import Operation, OpResult, Operand, Annotated, AnyAttr, Attribute, irdl_op_definition, \
    irdl_attr_definition, OptOpAttr
from xdsl.dialects.builtin import IntegerType, i64, Signedness, IntegerAttr, f64, AnyFloatAttr, AnyIntegerAttr
from xdsl.printer import Printer
from xdsl.dialects.memref import MemRefType, Alloc
from abc import ABC

printer = Printer(target=Printer.Target.MLIR)

t_uint32: IntegerType = IntegerType.from_width(32, Signedness.UNSIGNED)
t_int: IntegerType = IntegerType.from_width(32, Signedness.SIGNED)

AnyNumericAttr = AnyFloatAttr | AnyIntegerAttr


class MPIBaseOp(Operation, ABC):
    pass


@irdl_attr_definition
class Request(ParametrizedAttribute):
    name = 'mpi.request'


@irdl_attr_definition
class Status(ParametrizedAttribute):
    name = 'mpi.status'


@irdl_op_definition
class ISend(MPIBaseOp):
    name = 'mpi.isend'

    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]
    dest: Annotated[Operand, t_int]

    tag: OptOpAttr[IntegerAttr]

    request: Annotated[OpResult, Request()]

    @classmethod
    def get(cls, buff: Operand, dest: Operand, tag: int | None):
        attrs = {}

        if tag is not None:
            attrs['tag'] = IntegerAttr.from_params(tag, t_int)

        return cls.build(
            operands=[buff, dest],
            attributes=attrs,
            result_types=[Request()]
        )


@irdl_op_definition
class IRecv(MPIBaseOp):
    name = "mpi.irecv"

    source: Annotated[Operand, t_int]

    tag: OptOpAttr[IntegerAttr]

    buffer: Annotated[OpResult, MemRefType[AnyNumericAttr]]
    request: Annotated[OpResult, Request()]

    @classmethod
    def get(cls, source: Operand, dtype: MemRefType[AnyNumericAttr], tag: int | None = None):
        attrs = {}

        if tag is not None:
            attrs['tag'] = IntegerAttr.from_params(tag, t_int)

        return cls.build(
            operands=[source],
            attributes=attrs,
            result_types=[dtype, Request()]
        )


if __name__ == '__main__':
    reg = Region.from_operation_list([
        memref := Alloc.get(f64, 32, [100, 14, 14]),
        dest := Constant.from_int_and_width(1, t_int),
        req := ISend.get(memref, dest, 1),
        res := IRecv.get(dest, memref.results[0].typ, 1)
    ])

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
