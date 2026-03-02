from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, memref, mpi
from xdsl.dialects.builtin import (
    IndexType,
    IntegerType,
    MemRefType,
    Signedness,
    i32,
    i64,
)
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class MpiLibraryInfo:
    """
    This object is meant to capture characteristics of a specific MPI implementations.

    It holds magic values, sizes of structs, field offsets and much more.

    We need these as we currently cannot load these library headers into the programs we want to lower,
    therefore we need to generate our own external stubs and load magic values directly.

    This way of doing it is inherently fragile, but we don't know of any better way.
    We plan to include a C file that automagically extracts all this information from MPI headers.
    You can see the current C file used in this PR: https://github.com/xdslproject/xdsl/pull/526
    You can see the status of OpenMPI support here: https://github.com/xdslproject/xdsl/issues/523

    These defaults have been extracted from MPICH 3.3a2. We would highly suggest
    running the mpi-info.c file yourself with your version of the library!
    """

    # MPI_Datatype
    MPI_Datatype_size: int = 4
    MPI_CHAR: int = 0x4C000101
    MPI_SIGNED_CHAR: int = 0x4C000118
    MPI_UNSIGNED_CHAR: int = 0x4C000102
    MPI_BYTE: int = 0x4C00010D
    MPI_WCHAR: int = 0x4C00040E
    MPI_SHORT: int = 0x4C000203
    MPI_UNSIGNED_SHORT: int = 0x4C000204
    MPI_INT: int = 0x4C000405
    MPI_UNSIGNED: int = 0x4C000406
    MPI_LONG: int = 0x4C000807
    MPI_UNSIGNED_LONG: int = 0x4C000808
    MPI_FLOAT: int = 0x4C00040A
    MPI_DOUBLE: int = 0x4C00080B
    MPI_LONG_DOUBLE: int = 0x4C00100C
    MPI_LONG_LONG_INT: int = 0x4C000809
    MPI_UNSIGNED_LONG_LONG: int = 0x4C000819
    MPI_LONG_LONG: int = 0x4C000809

    # MPI_Op
    MPI_Op_size: int = 4
    MPI_MAX: int = 0x58000001
    MPI_MIN: int = 0x58000002
    MPI_SUM: int = 0x58000003
    MPI_PROD: int = 0x58000004
    MPI_LAND: int = 0x58000005
    MPI_BAND: int = 0x58000006
    MPI_LOR: int = 0x58000007
    MPI_BOR: int = 0x58000008
    MPI_LXOR: int = 0x58000009
    MPI_BXOR: int = 0x5800000A
    MPI_MINLOC: int = 0x5800000B
    MPI_MAXLOC: int = 0x5800000C
    MPI_REPLACE: int = 0x5800000D
    MPI_NO_OP: int = 0x5800000E

    # MPI_Comm
    MPI_Comm_size: int = 4
    MPI_COMM_WORLD: int = 0x44000000
    MPI_COMM_SELF: int = 0x44000001

    # MPI_Request
    MPI_Request_size: int = 4
    MPI_REQUEST_NULL = 0x2C000000

    # MPI_Status
    MPI_Status_size: int = 20
    MPI_STATUS_IGNORE: int = 0x00000001
    MPI_STATUSES_IGNORE: int = 0x00000001
    MPI_Status_field_MPI_SOURCE: int = (
        8  # offset of field MPI_SOURCE in struct MPI_Status
    )
    MPI_Status_field_MPI_TAG: int = 12  # offset of field MPI_TAG in struct MPI_Status
    MPI_Status_field_MPI_ERROR: int = (
        16  # offset of field MPI_ERROR in struct MPI_Status
    )

    # In place MPI All reduce
    MPI_IN_PLACE: int = -1


@dataclass
class _MPIToLLVMRewriteBase(RewritePattern, ABC):
    """
    This base is used to convert a pure rewrite function `lower` into the impure
    `match_and_rewrite` method used by xDSL.

    In order to lower that far, we require some information about the targeted MPI library
    (magic values, struct sizes, field offsets, etc.). This information is provided using
    the MpiLibraryInfo class.
    """

    MPI_SYMBOL_NAMES = {
        "mpi.init": "MPI_Init",
        "mpi.finalize": "MPI_Finalize",
        "mpi.irecv": "MPI_Irecv",
        "mpi.isend": "MPI_Isend",
        "mpi.wait": "MPI_Wait",
        "mpi.waitall": "MPI_Waitall",
        "mpi.comm.rank": "MPI_Comm_rank",
        "mpi.comm.size": "MPI_Comm_size",
        "mpi.recv": "MPI_Recv",
        "mpi.send": "MPI_Send",
        "mpi.reduce": "MPI_Reduce",
        "mpi.allreduce": "MPI_Allreduce",
        "mpi.bcast": "MPI_Bcast",
        "mpi.gather": "MPI_Gather",
    }
    """
    Translation table for mpi operation names to their MPI library function names
    """

    info: MpiLibraryInfo
    """
    This object carries information about the targeted MPI libraray
    """

    # Helpers
    def _get_mpi_dtype_size(
        self, mpi_dialect_dtype: mpi.RequestType | mpi.StatusType | mpi.DataType
    ):
        """
        This function retrieves the data size of a provided MPI type object
        """
        match mpi_dialect_dtype:
            case mpi.RequestType():
                return self.info.MPI_Request_size
            case mpi.StatusType():
                return self.info.MPI_Status_size
            case mpi.DataType():
                return self.info.MPI_Datatype_size

    def _emit_mpi_status_objs(
        self, number_to_output: int
    ) -> tuple[list[Operation], list[SSAValue | None], Operation]:
        """
        This function create operations that instantiate a pointer to an MPI_Status-sized object.

        If mpi_status_none = True is passed, it instead loads the magic value MPI_STATUS_IGNORE

        This is currently OpenMPI specific code, as other implementations probably have a different
        magic value for MPI_STATUS_NONE.
        """
        if number_to_output == 0:
            return (
                [
                    lit1 := arith.ConstantOp.from_int_and_width(1, builtin.i64),
                    res := llvm.IntToPtrOp(lit1),
                ],
                [],
                res,
            )
        else:
            return (
                [
                    lit1 := arith.ConstantOp.from_int_and_width(
                        number_to_output, builtin.i64
                    ),
                    res := llvm.AllocaOp(
                        lit1,
                        builtin.IntegerType(8 * self.info.MPI_Status_size),
                    ),
                ],
                [res.res],
                res,
            )

    def _emit_memref_counts(
        self, ssa_val: SSAValue
    ) -> tuple[list[Operation], OpResult]:
        """
        This takes in an SSA Value holding a memref, and creates operations
        to calculate the number of elements in the memref.

        It then returns a list of operations calculating that size, and
        an OpResult containing the calculated value.
        """
        assert isinstance(ssa_val_type := ssa_val.type, memref.MemRefType)

        # Note: we only allow MemRef, not UnrankedMemRef!
        # TODO: handle -1 in sizes
        if not all(dim >= 0 for dim in ssa_val_type.get_shape()):
            raise RuntimeError("MPI lowering does not support unknown-size memrefs!")

        size = prod(ssa_val_type.get_shape())

        literal = arith.ConstantOp.from_int_and_width(size, i32)
        return [literal], literal.result

    def _emit_mpi_operation_load(self, op_attr: mpi.OperationType) -> Operation:
        """
        This emits an instruction loading the correct magic MPI value for the
        operation into an SSA Value.
        """
        return arith.ConstantOp.from_int_and_width(
            self._translate_to_mpi_op(op_attr), i32
        )

    def _translate_to_mpi_op(self, op_attr: mpi.OperationType) -> int:
        """
        Translates an MPI dialect operation to the corresponding numeric value
        required by the underlying MPI library
        """
        if hasattr(self.info, op_attr.op_str.data):
            return getattr(self.info, op_attr.op_str.data)
        else:
            raise RuntimeError("Unknown MPI operation type")

    def _emit_mpi_type_load(self, type_attr: Attribute) -> Operation:
        """
        This emits an instruction loading the correct magic MPI value for the
        xDSL type of <type_attr> into an SSA Value.
        """
        return arith.ConstantOp.from_int_and_width(
            self._translate_to_mpi_type(type_attr), i32
        )

    def _translate_to_mpi_type(self, mpi_type: Attribute) -> int:
        """
        This translates an xDSL type to a corresponding MPI type

        Currently supported mappings are:
            floats:
                f32     -> MPI_FLOAT
                f64     -> MPI_DOUBLE
            ints:
                [u]i8   -> MPI_[UNSIGNED]_CHAR
                [u]i16  -> MPI_[UNSIGNED]_SHORT
                [u]i32  -> MPI_UNSIGNED / MPI_INT
                [u]i64  -> MPI_UNSIGNED_LONG_LONG / MPI_LONG_LONG_INT
        """
        if isinstance(mpi_type, builtin.Float32Type):
            return self.info.MPI_FLOAT
        if isinstance(mpi_type, builtin.Float64Type):
            return self.info.MPI_DOUBLE
        if isa(mpi_type, IntegerType):
            width: int = mpi_type.width.data
            if mpi_type.signedness.data == Signedness.UNSIGNED:
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
                f"MPI Datatype Conversion: Unsupported integer bitwidth: {width}"
            )
        raise ValueError(f"MPI Datatype Conversion: Unsupported type {mpi_type}")

    def _mpi_name(self, op: mpi.MPIBaseOp) -> str:
        """
        Convert the name of an mpi dialect operation to the corresponding MPI function call
        """
        if op.name not in self.MPI_SYMBOL_NAMES:
            raise RuntimeError(
                f"Lowering of MPI Operations failed, missing lowering for {op.name}!"
            )
        return self.MPI_SYMBOL_NAMES[op.name]

    def _memref_get_llvm_ptr(self, ref: SSAValue) -> tuple[list[Operation], Operation]:
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
            i64 := arith.IndexCastOp(index, builtin.i64),
            ptr := llvm.IntToPtrOp(i64),
        ], ptr


class LowerMpiInit(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.InitOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.InitOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        We currently don't model any argument passing to `MPI_Init()` and pass two nullptrs.
        """
        return [
            nullptr := llvm.ZeroOp(result_types=[llvm.LLVMPointerType()]),
            func.CallOp(self._mpi_name(op), [nullptr, nullptr], [i32]),
        ], []


class LowerMpiFinalize(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.FinalizeOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.FinalizeOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Relatively straight forward lowering of mpi.finalize operation.
        """
        return [
            func.CallOp(self._mpi_name(op), [], [i32]),
        ], []


class LowerMpiWait(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.WaitOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.WaitOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Relatively straight forward lowering of mpi.wait operation.
        """
        ops, new_results, res = self._emit_mpi_status_objs(len(op.results))
        return [
            *ops,
            func.CallOp(self._mpi_name(op), [op.request, res], [i32]),
        ], new_results


class LowerMpiWaitall(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.WaitallOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.WaitallOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Relatively straight forward lowering of mpi.waitall operation.
        """

        ops, new_results, res = self._emit_mpi_status_objs(len(op.results))
        return [
            *ops,
            func.CallOp(self._mpi_name(op), [op.count, op.requests, res], [i32]),
        ], new_results


class LowerMpiReduce(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.ReduceOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.ReduceOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Lowers the MPI Reduce operation
        """

        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            mpi_op := self._emit_mpi_operation_load(op.operationtype),
            func.CallOp(
                self._mpi_name(op),
                [
                    op.send_buffer,
                    op.recv_buffer,
                    op.count,
                    op.datatype,
                    mpi_op,
                    op.root,
                    comm_global,
                ],
                [],
            ),
        ], []


class LowerMpiAllreduce(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.AllreduceOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.AllreduceOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Lowers the MPI Allreduce operation
        """

        # Send buffer is optional (if not provided then call using MPI_IN_PLACE)
        has_send_buffer = op.send_buffer is not None

        comm_global = arith.ConstantOp.from_int_and_width(self.info.MPI_COMM_WORLD, i32)
        mpi_op = self._emit_mpi_operation_load(op.operationtype)

        operations = [comm_global, mpi_op]

        send_buffer_op: SSAValue | Operation
        if has_send_buffer:
            assert op.send_buffer is not None
            send_buffer_op = op.send_buffer
        else:
            send_buffer_op = arith.ConstantOp.from_int_and_width(
                self.info.MPI_IN_PLACE, i64
            )
            operations.append(send_buffer_op)

        return [
            *operations,
            func.CallOp(
                self._mpi_name(op),
                [
                    send_buffer_op,
                    op.recv_buffer,
                    op.count,
                    op.datatype,
                    mpi_op,
                    comm_global,
                ],
                [],
            ),
        ], []


class LowerMpiBcast(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.BcastOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.BcastOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Lowers the MPI Bcast operation
        """

        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            func.CallOp(
                self._mpi_name(op),
                [op.buffer, op.count, op.datatype, op.root, comm_global],
                [],
            ),
        ], []


class LowerMpiIsend(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.IsendOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.IsendOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.isend

        int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
        """

        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            func.CallOp(
                self._mpi_name(op),
                [
                    op.buffer,
                    op.count,
                    op.datatype,
                    op.dest,
                    op.tag,
                    comm_global,
                    op.request,
                ],
                [i32],
            ),
        ], []


class LowerMpiIrecv(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.IrecvOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.IrecvOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.irecv operations

        int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request)
        """

        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            func.CallOp(
                self._mpi_name(op),
                [
                    op.buffer,
                    op.count,
                    op.datatype,
                    op.source,
                    op.tag,
                    comm_global,
                    op.request,
                ],
                [i32],
            ),
        ], []


class LowerMpiSend(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.SendOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.SendOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.send operations

        MPI_Send signature:

        int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
                 int tag, MPI_Comm comm)
        """

        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            func.CallOp(
                self._mpi_name(op),
                [op.buffer, op.count, op.datatype, op.dest, op.tag, comm_global],
                [i32],
            ),
        ], []


class LowerMpiRecv(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.RecvOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.RecvOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.recv operations

        MPI_Recv signature:

        int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
        """

        mpi_status_ops, new_results, status = self._emit_mpi_status_objs(
            len(op.results)
        )

        return [
            *mpi_status_ops,
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            func.CallOp(
                self._mpi_name(op),
                [
                    op.buffer,
                    op.count,
                    op.datatype,
                    op.source,
                    op.tag,
                    comm_global,
                    status,
                ],
                [i32],
            ),
        ], new_results


class LowerMpiUnwrapMemRefOp(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.UnwrapMemRefOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.UnwrapMemRefOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        count_ops, count_ssa_val = self._emit_memref_counts(op.ref)
        extract_ptr_ops, ptr = self._memref_get_llvm_ptr(op.ref)

        elem_type = cast(MemRefType[mpi.AnyNumericType], op.ref.type).element_type

        return [
            *extract_ptr_ops,
            *count_ops,
            dtype := mpi.GetDtypeOp(elem_type),
        ], [ptr.results[0], count_ssa_val, dtype.result]


class LowerMpiGetDtype(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.GetDtypeOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.GetDtypeOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        return [
            dtype := self._emit_mpi_type_load(op.dtype),
        ], [dtype.results[0]]


class LowerMpiAllocateType(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.AllocateTypeOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.AllocateTypeOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Allocation operation, allocates the required memory as an LLVM pointer
        """
        datatype_size = self._get_mpi_dtype_size(op.dtype)
        return [
            request := llvm.AllocaOp(op.count, builtin.IntegerType(8 * datatype_size)),
        ], [request.results[0]]


class LowerMpiVectorGet(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.VectorGetOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.VectorGetOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This lowers the get array at index MPI operation in the dialect. Converts
        the pointer to an integer and then increments this to find the correct
        location before going back to a pointer and setting this as the result
        """

        assert mpi.VectorWrappableConstr.verifies(op.result.type)
        assert isa(op.vect.type, llvm.LLVMPointerType)
        datatype_size = self._get_mpi_dtype_size(op.result.type)

        return [
            ptr_int := llvm.PtrToIntOp(op.vect, i64),
            lit1 := arith.ConstantOp.from_int_and_width(datatype_size, 64),
            idx_cast1 := arith.IndexCastOp(op.element, IndexType()),
            idx_cast2 := arith.IndexCastOp(idx_cast1, i64),
            mul := arith.MuliOp(lit1, idx_cast2),
            add := arith.AddiOp(mul, ptr_int),
            out_ptr := llvm.IntToPtrOp(add),
        ], [out_ptr.results[0]]


class LowerMpiCommRank(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.CommRankOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.CommRankOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.comm.rank operation

        int MPI_Comm_rank(MPI_Comm comm, int *rank)
        """
        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            lit1 := arith.ConstantOp.from_int_and_width(1, 64),
            int_ptr := llvm.AllocaOp(lit1, i32),
            func.CallOp(self._mpi_name(op), [comm_global, int_ptr], [i32]),
            rank := llvm.LoadOp(int_ptr, IntegerType(32)),
        ], [rank.dereferenced_value]


class LowerMpiCommSize(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.CommSizeOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.CommSizeOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.comm.size operation

        int MPI_Comm_size(MPI_Comm comm, int *size)
        """
        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            lit1 := arith.ConstantOp.from_int_and_width(1, 64),
            int_ptr := llvm.AllocaOp(lit1, i32),
            func.CallOp(self._mpi_name(op), [comm_global, int_ptr], [i32]),
            rank := llvm.LoadOp(int_ptr, IntegerType(32)),
        ], [rank.dereferenced_value]


def add_external_func_defs(module: builtin.ModuleOp):
    """
    This rewriter adds all external function definitions for MPI calls to the module.

    It does so by first walking the whole module to discover MPI_ calls. Then
    it inserts a `func.Func.external()` op with the correct types at the end of the module.

    Make sure to apply this *in a separate pass after the lowerings*, otherwise
    this will match first and find no inserted MPI calls.
    """

    mpi_func_call_names = set(_MPIToLLVMRewriteBase.MPI_SYMBOL_NAMES.values())

    # collect all func calls to MPI functions
    funcs_to_emit: dict[str, tuple[Sequence[Attribute], Sequence[Attribute]]] = dict()

    for op in module.walk():
        if not isinstance(op, func.CallOp):
            continue
        if op.callee.string_value() not in mpi_func_call_names:
            continue
        funcs_to_emit[op.callee.string_value()] = (
            op.arguments.types,
            op.result_types,
        )

    # for each func found, add a FuncOp to the top of the module.
    for name, types in funcs_to_emit.items():
        SymbolTable.insert_or_update(module, func.FuncOp.external(name, *types))


class LowerNullRequestOp(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.NullRequestOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(
        self, op: mpi.NullRequestOp
    ) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.comm.size operation

        int MPI_Comm_size(MPI_Comm comm, int *size)
        """
        assert isa(op.request.type, llvm.LLVMPointerType)
        return [
            val := arith.ConstantOp.from_int_and_width(self.info.MPI_REQUEST_NULL, i32),
            llvm.StoreOp(val, op.request),
        ], []


class LowerMpiGatherOp(_MPIToLLVMRewriteBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.GatherOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, *self.lower(op))

    def lower(self, op: mpi.GatherOp) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.gather operation.


        int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       int root,
                       MPI_Comm comm)
        """
        return [
            comm_global := arith.ConstantOp.from_int_and_width(
                self.info.MPI_COMM_WORLD, i32
            ),
            func.CallOp(
                self._mpi_name(op),
                [
                    op.sendbuf,
                    op.sendcount,
                    op.sendtype,
                    op.recvbuf,
                    op.recvcount,
                    op.recvtype,
                    op.root,
                    comm_global,
                ],
                [i32],
            ),
        ], []


@dataclass(frozen=True)
class LowerMPIPass(ModulePass):
    name = "lower-mpi"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # TODO: how to get the lib info in here?
        lib_info = MpiLibraryInfo()
        walker1 = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerMpiInit(lib_info),
                    LowerMpiFinalize(lib_info),
                    LowerMpiWait(lib_info),
                    LowerMpiWaitall(lib_info),
                    LowerMpiCommRank(lib_info),
                    LowerMpiCommSize(lib_info),
                    LowerMpiIsend(lib_info),
                    LowerMpiIrecv(lib_info),
                    LowerMpiSend(lib_info),
                    LowerMpiRecv(lib_info),
                    LowerMpiReduce(lib_info),
                    LowerMpiAllreduce(lib_info),
                    LowerMpiBcast(lib_info),
                    LowerMpiUnwrapMemRefOp(lib_info),
                    LowerMpiGetDtype(lib_info),
                    LowerMpiAllocateType(lib_info),
                    LowerNullRequestOp(lib_info),
                    LowerMpiVectorGet(lib_info),
                    LowerMpiGatherOp(lib_info),
                ]
            ),
            apply_recursively=True,
        )

        walker1.rewrite_module(op)

        # add func.func to declare external functions
        add_external_func_defs(op)
