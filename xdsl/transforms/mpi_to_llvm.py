import dataclasses
from abc import ABC
from typing import TypeVar, cast

from xdsl.dialects.builtin import Signedness, IntegerType
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext

from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, op_type_rewrite_pattern, PatternRewriteWalker, \
    AnonymousRewritePattern, GreedyRewritePatternApplier
from xdsl.dialects import mpi, llvm, func, memref, arith, builtin


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
    MPI_UNSIGNED_CHAR: int = -1
    MPI_UNSIGNED_SHORT: int = -1
    MPI_UNSIGNED_LONG_LONG: int = -1
    MPI_CHAR: int = -1
    MPI_SHORT: int = -1
    MPI_LONG_LONG_INT: int = -1

    MPI_STATUS_IGNORE: int = 1

    request_size: int = 4
    status_size: int = 4 * 5
    mpi_comm_size: int = 4


_RewriteT = TypeVar('_RewriteT', bound=mpi.MPIBaseOp)


class _MPIToLLVMRewriteBase(RewritePattern, ABC):
    """
    This base is used to convert a pure rewrite function `lower` into the impure
    `match_and_rewrite` method used by xDSL.

    In order to lower that far, we require some information about the targeted MPI library
    (magic values, struct sizes, field offsets, etc.). This information is provided using
    the MpiLibraryInfo class.
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

    info: MpiLibraryInfo
    """
    This object carries information about the targeted MPI libraray
    """

    def __init__(self, info: MpiLibraryInfo):
        self.info = info

    # Helpers

    def _emit_mpi_status_obj(
        self, mpi_status_none: bool
    ) -> tuple[list[Operation], list[SSAValue | None], Operation]:
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

        literal = arith.Constant.from_int_and_width(size, mpi.t_int)
        return [literal], literal.result

    def _emit_mpi_type_load(self, type_attr: Attribute) -> Operation:
        """
        This emits an instruction loading the correct magic MPI value for the
        xDSL type of <type_attr> into an SSA Value.
        """
        return arith.Constant.from_int_and_width(
            self._translate_to_mpi_type(type_attr), mpi.t_int)

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

    def _mpi_name(self, op: mpi.MPIBaseOp) -> str:
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


class LowerMpiInit(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.Init, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(self,
              op: mpi.Init) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Relatively easy lowering of mpi.init operation.

        We currently don't model any argument passing to `MPI_Init()` and pass two nullptrs.
        """
        return [
            nullptr := llvm.NullOp.get(),
            func.Call.get(self._mpi_name(op), [nullptr, nullptr], [mpi.t_int]),
        ], []


class LowerMpiFinalize(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.Finalize, rewriter: PatternRewriter,
                          /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(
            self,
            op: mpi.Finalize) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Relatively easy lowering of mpi.finalize operation.
        """
        return [
            func.Call.get(self._mpi_name(op), [], [mpi.t_int]),
        ], []


class LowerMpiWait(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.Wait, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(self,
              op: mpi.Wait) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        Relatively easy lowering of mpi.wait operation.
        """
        ops, new_results, res = self._emit_mpi_status_obj(len(op.results) == 0)
        return [
            *ops,
            func.Call.get(self._mpi_name(op), [op.request, res], [mpi.t_int]),
        ], new_results


class LowerMpiISend(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.ISend, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(self,
              op: mpi.ISend) -> tuple[list[Operation], list[SSAValue | None]]:
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
        assert isinstance(op.buffer.typ, memref.MemRefType)
        memref_elm_typ = cast(memref.MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              mpi.t_int),
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data,
                                                     mpi.t_int),
            lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
            request := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.request_size)),
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.dest, tag, comm_global,
                request
            ], [mpi.t_int]),
        ], [request.results[0]]


class LowerMpiIRecv(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.IRecv, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(self,
              op: mpi.IRecv) -> tuple[list[Operation], list[SSAValue | None]]:
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
        assert isinstance(op.buffer.typ, memref.MemRefType)
        memref_elm_typ = cast(memref.MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data,
                                                     mpi.t_int),
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              mpi.t_int),
            lit1 := arith.Constant.from_int_and_width(1, builtin.i64),
            request := llvm.AllocaOp.get(
                lit1,
                builtin.IntegerType.from_width(8 * self.info.request_size)),
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.source, tag, comm_global,
                request
            ], [mpi.t_int]),
        ], [request.res]


class LowerMpiCommRank(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.CommRank, rewriter: PatternRewriter,
                          /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(
            self,
            op: mpi.CommRank) -> tuple[list[Operation], list[SSAValue | None]]:
        """
        This method lowers mpi.comm.rank operation

        int MPI_Comm_rank(MPI_Comm comm, int *rank)
        """
        return [
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              mpi.t_int),
            lit1 := arith.Constant.from_int_and_width(1, 64),
            int_ptr := llvm.AllocaOp.get(lit1, mpi.t_int),
            func.Call.get(self._mpi_name(op), [comm_global, int_ptr],
                          [mpi.t_int]),
            rank := llvm.LoadOp.get(int_ptr),
        ], [rank.dereferenced_value]


class LowerMpiSend(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.Send, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(self,
              op: mpi.Send) -> tuple[list[Operation], list[SSAValue | None]]:
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
        assert isinstance(op.buffer.typ, memref.MemRefType)
        memref_elm_typ = cast(memref.MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data,
                                                     mpi.t_int),
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              mpi.t_int),
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            func.Call.get(
                self._mpi_name(op),
                [ptr[1], count_ssa_val, datatype, op.dest, tag, comm_global],
                [mpi.t_int]),
        ], []


class LowerMpiRecv(_MPIToLLVMRewriteBase):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.Recv, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(*self.lower(op))

    def lower(self,
              op: mpi.Recv) -> tuple[list[Operation], list[SSAValue | None]]:
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
        assert isinstance(op.buffer.typ, memref.MemRefType)
        memref_elm_typ = cast(memref.MemRefType[Attribute],
                              op.buffer.typ).element_type

        return [
            *count_ops,
            *ops,
            *(ptr := self._memref_get_llvm_ptr(op.buffer))[0],
            datatype := self._emit_mpi_type_load(memref_elm_typ),
            tag := arith.Constant.from_int_and_width(op.tag.value.data,
                                                     mpi.t_int),
            comm_global :=
            arith.Constant.from_int_and_width(self.info.mpi_comm_world_val,
                                              mpi.t_int),
            func.Call.get(self._mpi_name(op), [
                ptr[1], count_ssa_val, datatype, op.source, tag, comm_global,
                status
            ], [mpi.t_int]),
        ], new_results

    # Miscellaneous


class MpiAddExternalFuncDefs(RewritePattern):
    mpi_func_call_names = set(_MPIToLLVMRewriteBase.MPI_SYMBOL_NAMES.values())

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp,
                          rewriter: PatternRewriter, /):
        # collect all func calls to MPI functions
        funcs_to_emit: dict[str, tuple[list[Attribute],
                                       list[Attribute]]] = dict()

        @op_type_rewrite_pattern
        def match_func(op: func.Call, rewriter: PatternRewriter, /):
            if op.callee.string_value() not in self.mpi_func_call_names:
                return
            funcs_to_emit[op.callee.string_value()] = (list(
                arg.typ
                for arg in op.arguments), list(res.typ for res in op.results))

        rewriter=PatternRewriteWalker(AnonymousRewritePattern(match_func))
        rewriter.rewrite_module(module)

        # for each func found, add a FuncOp to the top of the module.
        for name, types in funcs_to_emit.items():
            arg, res = types
            rewriter.insert_op_at_pos(func.FuncOp.external(name, arg, res),
                                      module.body.blocks[0],
                                      len(module.body.blocks[0].ops))


def mpi_to_llvm_lowering(ctx: MLContext, module: builtin.ModuleOp):
    # TODO: how to get the lib info in here?
    lib_info = MpiLibraryInfo()

    # lower to func.call
    walker1 = PatternRewriteWalker(
        GreedyRewritePatternApplier([
            LowerMpiInit(lib_info),
            LowerMpiFinalize(lib_info),
            LowerMpiWait(lib_info),
            LowerMpiISend(lib_info),
            LowerMpiIRecv(lib_info),
            LowerMpiCommRank(lib_info),
            LowerMpiSend(lib_info),
            LowerMpiRecv(lib_info),
        ]))
    walker1.rewrite_module(module)

    # add func.func to declare external functions
    walker2 = PatternRewriteWalker(MpiAddExternalFuncDefs())
    walker2.rewrite_module(module)
