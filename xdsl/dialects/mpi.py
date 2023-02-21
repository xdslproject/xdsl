import dataclasses
from abc import ABC
from enum import Enum

from xdsl.dialects import llvm
from xdsl.dialects import builtin, arith, memref, func
from xdsl.dialects.builtin import (IntegerType, Signedness, IntegerAttr,
                                   AnyFloatAttr, AnyIntegerAttr, StringAttr)
from xdsl.dialects.memref import MemRefType, Alloc
from xdsl.ir import OpResult, ParametrizedAttribute, Dialect, MLIRType
from xdsl.ir import Operation, Attribute, SSAValue
from xdsl.irdl import (Operand, Annotated, irdl_op_definition,
                       irdl_attr_definition, OptOpAttr, OpAttr, OptOpResult)

from xdsl.pattern_rewriter import PatternRewriter, RewritePattern

f64 = builtin.f64

t_int: IntegerType = IntegerType.from_width(32, Signedness.SIGNLESS)
t_bool: IntegerType = IntegerType.from_width(1, Signedness.SIGNLESS)

AnyNumericAttr = AnyFloatAttr | AnyIntegerAttr


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
    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]

    tag: OptOpAttr[AnyIntegerAttr]

    request: Annotated[OpResult, RequestType()]

    @classmethod
    def get(cls,
            source: Operand,
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

        - We bundle buf, count and datatype into the type definition and use `memref`
        - We assume this type information is compile-time known
        - We assume tag is compile-time known
        - We omit the possibility of using multiple communicators
    """

    name = "mpi.recv"

    source: Annotated[Operand, t_int]
    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]

    tag: OptOpAttr[AnyIntegerAttr]

    status: Annotated[OptOpResult, StatusType]

    @classmethod
    def get(cls,
            source: Operand,
            dtype: MemRefType[AnyNumericAttr],
            tag: int | None = None,
            ignore_status: bool = True):
        return cls.build(operands=[source],
                         attributes=_build_attr_dict_with_optional_tag(tag),
                         result_types=[dtype] +
                         ([[]] if ignore_status else [[StatusType()]]))


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
    status: Annotated[OptOpResult, t_int]

    @classmethod
    def get(cls, request: Operand, ignore_status: bool = True):
        if ignore_status:
            result_types = [[]]
        else:
            result_types = [[t_int]]

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

    result: Annotated[OpResult, t_int]

    @classmethod
    def get(cls, status_obj: Operand, field: StatusTypeField):
        return cls.build(operands=[status_obj],
                         attributes={'field': StringAttr(field.value)},
                         result_types=[t_int])


@irdl_op_definition
class CommRank(MPIBaseOp):
    name = "mpi.comm.rank"

    rank: Annotated[OpResult, t_int]

    @classmethod
    def get(cls):
        return cls.build(result_types=[t_int])


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


@dataclasses.dataclass
class MpiLibraryInfo:
    mpi_comm_world_val: int = 0x44000000

    MPI_INT: int = 0x4c000405
    MPI_UNSIGNED: int = 0x4c000406
    MPI_LONG: int = 0x4c000807
    MPI_UNSIGNED_LONG: int = 0x4c000808
    MPI_FLOAT: int = 0x4c00040a
    MPI_DOUBLE: int = 0x4c00080b
    MPI_STATUS_IGNORE: int = 1

    request_size: int = 4
    status_size: int = 4
    mpi_comm_size: int = 4


class MpiLowerings(RewritePattern):
    _emitted_function_calls: dict[str, tuple[list[Attribute], list[Attribute]]]

    MPI_SYMBOL_NAMES = {
        'mpi.init': 'MPI_Init',
        'mpi.finalize': 'MPI_Finalize',
        'mpi.irecv': 'MPI_Irecv',
        'mpi.isend': 'MPI_Isend',
        'mpi.wait': 'MPI_Wait',
        'mpi.comm.rank': 'MPI_Comm_rank',
    }

    def __init__(self, info: MpiLibraryInfo):
        self.info = info
        self._emitted_function_calls = dict()

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, MPIBaseOp):
            return

        field = "lower_{}".format(op.name.replace('.', '_'))

        if hasattr(self, field):
            new_ops, *other = getattr(self, field)(op)
            rewriter.replace_matched_op(new_ops, *other)

            for op in new_ops:
                if not isinstance(op, func.Call) or not op.callee.string_value(
                ).startswith('MPI_'):
                    continue
                self._emitted_function_calls[op.callee.string_value()] = (
                    [x.typ for x in op.arguments],
                    [x.typ for x in op.results],
                )
        else:
            print("Missing lowering for {}".format(op.name))

    # Individual lowerings:

    def lower_mpi_init(self, op: Init):
        # and then we emit a func.call op
        return [
            nullptr := llvm.NullOp.get(t_int),
            func.Call.get(self._mpi_name(op), [nullptr, nullptr], [t_int]),
        ], []

    def lower_mpi_finalize(self, op: Init):
        return [
            func.Call.get(self._mpi_name(op), [], [t_int]),
        ], []

    def lower_mpi_wait(self, op: Wait):
        if len(op.status.uses) == 0:
            pass
            # TODO: emit MPI_STATUS_IGNORE

        return [
            lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
            res := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.status_size)
            ),
            func.Call.get(self._mpi_name(op), [op.request, res], [t_int])
        ], [res]  # yapf: disable

    def lower_mpi_isend(self, op: ISend):
        """
        int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
        """
        count_ops, count_ssa_val = self._emit_memref_counts(op.buffer)
        # TODO: correctly infer mpi type from memref

        return [
            *count_ops,
            datatype := arith.Constant.from_int_and_width(self.info.MPI_DOUBLE, t_int),
            tag := arith.Constant.from_int_and_width(op.tag.value.data, t_int),
            comm_global := arith.Constant.from_int_and_width(self.info.mpi_comm_world_val, t_int),
            lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
            request := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.request_size)
            ),
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.dest, tag, comm_global,
                request
            ], [t_int])
        ], [request.results[0]]  # yapf: disable

    def lower_mpi_irecv(self, op: IRecv):
        """
        int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request)
        """
        count_ops, count_ssa_val = self._emit_memref_counts(op.buffer)

        return [
            *count_ops,
            buffer      := memref.Alloc.get(op.buffer.typ, 32),
            *(ptr       := self._memref_get_llvm_ptr(buffer))[0],
            datatype    := arith.Constant.from_int_and_width(self.info.MPI_DOUBLE, t_int),
            tag         := arith.Constant.from_int_and_width(op.tag.value.data, t_int),
            comm_global := arith.Constant.from_int_and_width(self.info.mpi_comm_world_val, t_int),
            lit1        := arith.Constant.from_int_and_width(1, builtin.i64),
            request     := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.request_size)
            ),
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.source, tag, comm_global,
                request
            ], [t_int])
        ], [buffer.memref, request.res] # yapf: disable

    def lower_mpi_comm_rank(self, op: CommRank):
        return [
            comm_global := arith.Constant.from_int_and_width(self.info.mpi_comm_world_val, t_int),
            lit1    := arith.Constant.from_int_and_width(1, 64),
            int_ptr := llvm.AllocaOp.get(lit1, t_int),
            func.Call.get(
                self._mpi_name(op),
                [comm_global, int_ptr],
                [t_int]
            ),
            rank    := llvm.LoadOp.get(int_ptr)
        ], [rank.dereferenced_value]  # yapf: disable

    # Miscellaneous

    def _emit_memref_counts(
            self, ssa_val: SSAValue) -> tuple[list[Operation], SSAValue]:
        # Note: we only allow MemRef, not UnrankedMemref!
        # TODO: handle -1 in sizes
        assert isinstance(ssa_val.typ, memref.MemRefType)
        size = sum(dim.value.data for dim in ssa_val.typ.shape.data)

        literal = arith.Constant.from_int_and_width(size, builtin.i64)
        return [literal], literal.result

    def _mpi_name(self, op):
        if op.name not in self.MPI_SYMBOL_NAMES:
            print("unknown MPI op:  {}".format(op.name))
        return self.MPI_SYMBOL_NAMES[op.name]

    def _emit_external_funcs(self):
        return [
            func.FuncOp.external(name, *args)
            for name, args in self._emitted_function_calls.items()
        ]

    def _memref_get_llvm_ptr(self, ref: SSAValue):
        """
          %0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
          %1 = arith.index_cast %0 : index to i64
          %2 = llvm.inttoptr %1 : i64 to !llvm.ptr<f32>
        """
        return [
            index := memref.ExtractAlignedPointerAsIndexOp.get(ref),
            i64 := arith.IndexCastOp.get(index, builtin.i64),
            ptr := llvm.IntToPtrOp.get(i64)
        ], ptr  # yapf: disable

    def _alloc_type(self, size: int):
        return llvm.LLVMPointerType.typed(
            builtin.IntegerType.from_width(size * 8))

    def insert_externals_into_module(self, op: builtin.ModuleOp):
        for func in self._emit_external_funcs():
            op.regions[0].blocks[0].insert_op(func, 0)
