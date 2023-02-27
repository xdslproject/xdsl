from __future__ import annotations

import dataclasses
from abc import ABC
from enum import Enum
from typing import cast

from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.dialects.builtin import (IntegerType, Signedness, IntegerAttr,
                                   AnyFloatAttr, AnyIntegerAttr, StringAttr)
from xdsl.dialects import builtin, arith, func, memref, llvm
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, Attribute, SSAValue, OpResult, ParametrizedAttribute, Dialect, MLIRType
from xdsl.irdl import (Operand, Annotated, irdl_op_definition,
                       irdl_attr_definition, OpAttr, OptOpResult)

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

    tag: OpAttr[IntegerAttr[Annotated[IntegerType, t_int]]]

    request: Annotated[OpResult, RequestType]

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

    tag: OpAttr[IntegerAttr[Annotated[IntegerType, t_int]]]

    @classmethod
    def get(cls, buff: SSAValue | Operation, dest: SSAValue | Operation,
            tag: int) -> Send:
        return cls.build(operands=[buff, dest],
                         attributes=_build_attr_dict_with_optional_tag(tag),
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

        - We bundle buf, count and datatype into the type definition and use `memref`
        - We assume tag is compile-time known
        - We omit the possibility of using multiple communicators
    """

    name = "mpi.irecv"

    source: Annotated[Operand, t_int]
    buffer: Annotated[Operand, MemRefType[AnyNumericAttr]]

    tag: OpAttr[IntegerAttr[Annotated[IntegerType, t_int]]]

    request: Annotated[OpResult, RequestType]

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

    tag: OpAttr[IntegerAttr[Annotated[IntegerType, t_int]]]

    status: Annotated[OptOpResult, StatusType]

    @classmethod
    def get(cls,
            source: SSAValue | Operation,
            buffer: SSAValue | Operation,
            tag: int | None = None,
            ignore_status: bool = True):
        return cls.build(
            operands=[source, buffer],
            attributes=_build_attr_dict_with_optional_tag(tag),
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
    status: Annotated[OptOpResult, t_int]

    @classmethod
    def get(cls, request: Operand, ignore_status: bool = True):
        result_types: list[list[Attribute]] = [[t_int]]
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
    """
    This represents a bare MPI_Init call with both args being nullptr
    """
    name = "mpi.init"


@irdl_op_definition
class Finalize(MPIBaseOp):
    name = "mpi.finalize"


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
], [
    RequestType,
    StatusType,
])


@dataclasses.dataclass
class MpiLibraryInfo:
    """
    This object is meant to capture characteristics of a specific MPI implementations.

    It holds magic values, sizes of structs, field offsets and much more.

    We need these as we currently cannot load these library headers into the programs we want to lower,
    therefore we need to generate our own external stubs and load magic values directly.

    This way of doing it is inherently fragile, but we don't know of any better way.
    We plan to include a C file that automagically extracts all this information from MPI headers.

    These defaults have been chosen to work with **our** version of OpenMPI. No guarantees of portability!
    """
    mpi_comm_world_val: int = 0x44000000

    MPI_INT: int = 0x4c000405
    MPI_UNSIGNED: int = 0x4c000406
    MPI_LONG: int = 0x4c000807
    MPI_UNSIGNED_LONG: int = 0x4c000808
    MPI_FLOAT: int = 0x4c00040a
    MPI_DOUBLE: int = 0x4c00080b
    MPI_UNSIGNED_CHAR = -1
    MPI_UNSIGNED_SHORT = -1
    MPI_UNSIGNED_LONG_LONG = -1
    MPI_CHAR = -1
    MPI_SHORT = -1
    MPI_LONG_LONG_INT = -1

    MPI_STATUS_IGNORE: int = 1

    request_size: int = 4
    status_size: int = 4 * 5
    mpi_comm_size: int = 4


class MpiLowerings(RewritePattern):
    """
    This rewrite pattern contains rules to lower the MPI dialect to llvm+func+builin

    In order to lower that far, we require some information about the targeted MPI library
    (magic values, struct sizes, field offsets, etc.). This information is provided using
    the MpiLibraryInfo class.
    """

    _emitted_function_calls: dict[str, tuple[list[Attribute], list[Attribute]]]
    """
    This object keeps track of all the functions we have emitted and their type signature.

    This is done so we can later on add "external" function declarations so LLVM is happy :)
    """

    MPI_SYMBOL_NAMES = {
        'mpi.init': 'MPI_Init',
        'mpi.finalize': 'MPI_Finalize',
        'mpi.irecv': 'MPI_Irecv',
        'mpi.isend': 'MPI_Isend',
        'mpi.wait': 'MPI_Wait',
        'mpi.comm.rank': 'MPI_Comm_rank',
        'mpi.recv': 'MPI_Recv',
        'mpi.send': 'MPI_Send'
    }
    """
    Translation table for mpi operation names to their MPI library function names
    """

    def __init__(self, info: MpiLibraryInfo):
        self.info = info
        self._emitted_function_calls = dict()

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        """
        This method acts as a dispatcher to lower individual MPI operations.

        It tries to dispatch calls for each mpi.<name> op to lower_mpi_<name> methods.

        The methods each return the argument inputs to rewriter.replace_matched_op calls
        """
        if not isinstance(op, MPIBaseOp):
            return

        field = "lower_{}".format(op.name.replace('.', '_'))

        if hasattr(self, field):
            new_ops, *other = getattr(self, field)(op)
            rewriter.replace_matched_op(new_ops, *other)

            for op in new_ops:
                if not isinstance(op, func.Call):
                    continue
                if not op.callee.string_value().startswith('MPI_'):
                    continue
                self._emitted_function_calls[op.callee.string_value()] = (
                    [x.typ for x in op.arguments],
                    [x.typ for x in op.results],
                )
        else:
            print("Missing lowering for {}".format(op.name))

    # Individual lowerings:

    def lower_mpi_init(self,
                       op: Init) -> tuple[list[Operation], list[OpResult]]:
        """
        Relatively easy lowering of mpi.init operation.

        We currently don't model any argument passing to `MPI_Init()` and pass two nullptrs.
        """
        return [
            nullptr := llvm.NullOp.get(),
            func.Call.get(self._mpi_name(op), [nullptr, nullptr], [t_int]),
        ], []

    def lower_mpi_finalize(self,
                           op: Finalize) -> tuple[list[Operation], list[OpResult]]:
        """
        Relatively easy lowering of mpi.finalize operation.
        """
        return [
            func.Call.get(self._mpi_name(op), [], [t_int]),
        ], []

    def lower_mpi_wait(self,
                       op: Wait) -> tuple[list[Operation], list[OpResult]]:
        """
        Relatively easy lowering of mpi.wait operation.
        """
        ops, new_results, res = self._emit_mpi_status_obj(len(op.results) == 0)
        return [
            *ops,
            func.Call.get(self._mpi_name(op), [op.request, res], [t_int]),
        ], new_results

    def lower_mpi_isend(self,
                        op: ISend) -> tuple[list[Operation], list[OpResult]]:
        """
        This method lowers mpi.isend

        int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
        """
        count_ops, count_ssa_val = self._emit_memref_counts(op.buffer)

        # TODO: I really hate this dance just to make pyright happy
        #       imo this makes code *less* readable.
        #       The _MemRefTypeElement is bound to Attribute, so
        #       op.buffer.typ.element_type is ALLWAYS at least Attribute!
        assert isinstance(op.buffer.typ, MemRefType)
        memref_elm_typ = cast(MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              t_int),
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data, t_int),
            lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
            request := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.request_size)),
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.dest, tag, comm_global,
                request
            ], [t_int]),
        ], [request.results[0]]

    def lower_mpi_irecv(self,
                        op: IRecv) -> tuple[list[Operation], list[OpResult]]:
        """
        This method lowers mpi.irecv operations

        int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request)
        """
        count_ops, count_ssa_val = self._emit_memref_counts(op.buffer)

        # TODO: I really hate this dance just to make pyright happy
        #       imo this makes code *less* readable.
        #       The _MemRefTypeElement is bound to Attribute, so
        #       op.buffer.typ.element_type is ALLWAYS at least Attribute!
        assert isinstance(op.buffer.typ, MemRefType)
        memref_elm_typ = cast(MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data, t_int),
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              t_int),
            lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
            request := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.request_size)),
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.source, tag, comm_global,
                request
            ], [t_int]),
        ], [request.res]

    def lower_mpi_comm_rank(
            self, op: CommRank) -> tuple[list[Operation], list[OpResult]]:
        """
        This method lowers mpi.comm.rank operation

        int MPI_Comm_rank(MPI_Comm comm, int *rank)
        """
        return [
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              t_int),
            lit1 := arith.Constant.from_int_and_width(1, 64),
            int_ptr := llvm.AllocaOp.get(lit1, t_int),
            func.Call.get(self._mpi_name(op), [comm_global, int_ptr], [t_int]),
            rank := llvm.LoadOp.get(int_ptr),
        ], [rank.dereferenced_value]

    def lower_mpi_send(self,
                       op: Send) -> tuple[list[Operation], list[OpResult]]:
        """
        This method lowers mpi.send operations

        MPI_Send signature:

        int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
                 int tag, MPI_Comm comm)
        """
        count_ops, count_ssa_val = self._emit_memref_counts(op.buffer)

        # TODO: I really hate this dance just to make pyright happy
        #       imo this makes code *less* readable.
        #       The _MemRefTypeElement is bound to Attribute, so
        #       op.buffer.typ.element_type is ALLWAYS at least Attribute!
        assert isinstance(op.buffer.typ, MemRefType)
        memref_elm_typ = cast(MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data, t_int),
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              t_int),
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            func.Call.get(
                self._mpi_name(op),
                [ptr[1], count_ssa_val, datatype, op.dest, tag, comm_global],
                [t_int]),
        ], []

    def lower_mpi_recv(self,
                       op: Recv) -> tuple[list[Operation], list[OpResult]]:
        """
        This method lowers mpi.recv operations

        MPI_Recv signature:

        int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
        """
        count_ops, count_ssa_val = self._emit_memref_counts(op.buffer)

        ops, new_results, status = self._emit_mpi_status_obj(
            len(op.results) == 0)

        # TODO: I really hate this dance just to make pyright happy
        #       imo this makes code *less* readable.
        #       The _MemRefTypeElement is bound to Attribute, so
        #       op.buffer.typ.element_type is ALLWAYS at least Attribute!
        assert isinstance(op.buffer.typ, MemRefType)
        memref_elm_typ = cast(MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            *ops,
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data, t_int),
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              t_int),
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.source, tag, comm_global,
                status
            ], [t_int]),
        ], new_results

    # Miscellaneous

    def _emit_mpi_status_obj(
        self, mpi_status_none: bool
    ) -> tuple[list[Operation], list[OpResult], Operation]:
        """
        This function create operations that instantiate a pointer to an MPI_Status-sized object.

        If mpi_status_none = True is passed, it instead loads the magic value MPI_STATUS_IGNORE

        This is currently OpenMPI specific code, as other implementations probably have a different
        magic value for MPI_STATUS_NONE.
        """
        if mpi_status_none:
            return [
                lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
                res := llvm.IntToPtrOp.get(lit1),
            ], [], res
        else:
            return [
                lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
                res := llvm.AllocaOp.get(lit1,
                                         builtin.IntegerType.from_width(
                                             8 * self.info.status_size),
                                         as_untyped_ptr=True),
            ], [res.res], res

    def _emit_memref_counts(
            self, ssa_val: SSAValue) -> tuple[list[Operation], OpResult]:
        """
        This takes in an SSA Value holding a memref, and creates operations
        to calculate the number of elements in the memref.

        It then returns a list of operations calculating that size, and
        an OpResult containing the calculated value.
        """
        assert isinstance(ssa_val.typ, memref.MemRefType)

        # Note: we only allow MemRef, not UnrankedMemref!
        # TODO: handle -1 in sizes
        if not all(dim.value.data >= 0 for dim in ssa_val.typ.shape.data):
            raise RuntimeError(
                "MPI lowering does not support unknown-size memrefs!")

        size = sum(dim.value.data for dim in ssa_val.typ.shape.data)

        literal = arith.Constant.from_int_and_width(size, t_int)
        return [literal], literal.result

    def _emit_mpi_type_load(self, type_attr: Attribute) -> Operation:
        """
        This emits an instruction loading the correct magic MPI value for the
        xDSL type of <type_attr> into an SSA Value.
        """
        return arith.Constant.from_int_and_width(
            self._translate_to_mpi_type(type_attr), t_int)

    def _translate_to_mpi_type(self, typ: Attribute) -> int:
        """
        This translates an xDSL type to a corresponding MPI type

        Currently supported mappings are:
            floats:
                f32     -> MPI_FLOAT
                f64     -> MPI_DOUBLR
            ints:
                [u]i8   -> MPI_[UNSIGNED]_CHAR
                [u]i16  -> MPI_[UNSIGNED]_SHORT
                [u]i32  -> MPI_UNSIGNED / MPI_INT
                [u]i64  -> MPI_UNSIGNED_LONG_LONG / MPI_LONG_LONG_INT
        """
        if isinstance(typ, builtin.Float32Type):
            return self.info.MPI_FLOAT
        if isinstance(typ, builtin.Float64Type):
            return self.info.MPI_DOUBLE
        if isinstance(typ, IntegerType):
            width: int = typ.width.data
            if typ.signedness.data == Signedness.UNSIGNED:
                # unsigned branch
                if width == 8:
                    return self.info.MPI_UNSIGNED_CHAR
                if width == 16:
                    return self.info.MPI_UNSIGNED_SHORT
                if width == 32:
                    return self.info.MPI_UNSIGNED
                if width == 64:
                    return self.info.MPI_UNSIGNED_LONG_LONG
            else:
                if width == 8:
                    return self.info.MPI_CHAR
                if width == 16:
                    return self.info.MPI_SHORT
                if width == 32:
                    return self.info.MPI_INT
                if width == 64:
                    return self.info.MPI_LONG_LONG_INT
            raise ValueError(
                "MPI Datatype Conversion: Unsupported integer bitwidth: {}".
                format(width))
        raise ValueError(
            "MPI Datatype Conversion: Unsupported type {}".format(typ))

    def _mpi_name(self, op: MPIBaseOp) -> str:
        """
        Convert the name of an mpi dialect operation to the corresponding MPI function call
        """
        if op.name not in self.MPI_SYMBOL_NAMES:
            raise RuntimeError(
                "Lowering of MPI Operations failed, missing lowering for {}!".
                format(op.name))
        return self.MPI_SYMBOL_NAMES[op.name]

    def _memref_get_llvm_ptr(
            self, ref: SSAValue) -> tuple[list[Operation], Operation]:
        """
        Converts an SSA Value holding a reference to a memref to llvm.ptr

        The official way as per the documentations pecifies the following
        sequence of operations:

          %0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
          %1 = arith.index_cast %0 : index to i64
          %2 = llvm.inttoptr %1 : i64 to !llvm.ptr<f32>

        https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_aligned_pointer_as_index-mlirmemrefextractalignedpointerasindexop
        """
        return [
            index := memref.ExtractAlignedPointerAsIndexOp.get(ref),
            i64 := arith.IndexCastOp.get(index, builtin.i64),
            ptr := llvm.IntToPtrOp.get(i64),
        ], ptr

    def _emit_external_funcs(self) -> list[Operation]:
        """
        This method generates external function definitions for all function calls to MPI
        libraries generated using this instance of the lowering rewrites.
        """
        return [
            func.FuncOp.external(name, *args)
            for name, args in self._emitted_function_calls.items()
        ]

    def insert_externals_into_module(self, op: builtin.ModuleOp):
        """
        This function inserts all external function definitions for MPI function at the top of
        the given module.

        This can only be called AFTER you applied this rewrite to your module, otherwise no
        external functions will be inserted!
        """
        for func_op in self._emit_external_funcs():
            op.regions[0].blocks[0].insert_op(func_op, 0)
