from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from xdsl.dialects import arith, builtin, llvm, memref, mpi
from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass(frozen=True)
class MpichLibraryInfo:
    """
    This object holds magic values of MPICH-style MPI implementations.

    It holds magic values, sizes of structs, field offsets and more.

    We need these as we currently cannot load these library headers into the programs we want to lower,
    therefore we need to generate our own external stubs and load magic values directly.
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


class MPITargetInfo(ABC):
    name: ClassVar[str]

    def __init__(self):
        pass

    @abstractmethod
    def materialize_type(self, type: Attribute) -> tuple[list[Operation], SSAValue]:
        """
        Converts an MLIR type to an MPI_Datatype constant.
        """
        raise NotImplementedError()

    @abstractmethod
    def globals_to_add(self) -> Iterable[Operation]:
        """
        Provides a list of global symbols to inject into the module.
        """
        raise NotImplementedError()

    @abstractmethod
    def materialize_named_constant(self, name: str) -> tuple[list[Operation], SSAValue]:
        """
        Takes the name of an MPI named symbol and returns a series of operations that
        instantiates the constant.
        """
        raise NotImplementedError()


class MpichTargetInfo(MPITargetInfo):
    name = "mpich"
    constants: MpichLibraryInfo = MpichLibraryInfo()

    def materialize_type(self, typ: Attribute):
        return [
            res := arith.Constant.from_int_and_width(
                self._type_to_magic_number(typ), 32
            )
        ], res.result

    def _type_to_magic_number(self, typ: Attribute):
        match typ:
            case builtin.i32:
                return self.constants.MPI_INT
            case builtin.i64:
                return self.constants.MPI_LONG_LONG
            case builtin.Float32Type():
                return self.constants.MPI_FLOAT
            case builtin.Float64Type():
                return self.constants.MPI_DOUBLE
            case unknown:
                raise ValueError(f"Cannot lower MPI type {unknown}")

    def materialize_named_constant(self, name: str) -> tuple[list[Operation], SSAValue]:
        return [
            res := arith.Constant.from_int_and_width(self.constants.__dict__[name], 32)
        ], res.result

    def globals_to_add(self) -> Iterable[Operation]:
        return []


class OpenMPITargetInfo(MPITargetInfo):
    name = "opnempi"
    constants: MpichLibraryInfo = MpichLibraryInfo()

    seen_globals: set[str]

    def __init__(self):
        super().__init__()
        self.seen_globals = set()

    def materialize_type(self, typ: Attribute):
        name = self._type_to_global_name(typ)
        self.seen_globals.add(name)
        return [
            res := llvm.AddressOfOp(name, llvm.LLVMPointerType.opaque())
        ], res.result

    def materialize_named_constant(self, name: str) -> tuple[list[Operation], SSAValue]:
        if name == "MPI_STATUS_IGNORE":
            return [res := llvm.NullOp()], res.nullptr
        name = "ompi_" + name.lower()
        self.seen_globals.add(name)
        return [
            res := llvm.AddressOfOp(name, llvm.LLVMPointerType.opaque())
        ], res.result

    def _type_to_global_name(self, typ: Attribute) -> str:
        match typ:
            case builtin.i32:
                return "ompi_mpi_int"
            case builtin.i32:
                return "ompi_mpi_long_long_int"
            case builtin.Float32Type():
                return "ompi_mpi_float"
            case builtin.Float64Type():
                return "ompi_mpi_double"
            case unknown:
                raise ValueError(f"Cannot lower MPI type {unknown}")

    def globals_to_add(self) -> Iterable[Operation]:
        return [
            llvm.GlobalOp(
                # use int as a placeholder
                builtin.i32,
                name,
                "external",
            )
            for name in sorted(self.seen_globals)
        ]


@dataclass
class MpiLoweringPattern(RewritePattern):
    target: MPITargetInfo

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mpi.MPIBaseOp, rewriter: PatternRewriter, /):
        match op:
            case mpi.InitOp():
                rewriter.replace_matched_op(self.lower_init(), [])
            case mpi.FinalizeOp():
                rewriter.replace_matched_op(self.lower_finalize(), [])
            case mpi.GetDtypeOp(dtype=typ):
                ops, res = self.target.materialize_type(typ)
                rewriter.replace_matched_op(ops, [res])
            case mpi.UnwrapMemrefOp() as unw:
                rewriter.replace_matched_op(*self.lower_unwrap_memref(unw))
            case mpi.SendOp() as send:
                assert isinstance(send, mpi.SendOp)
                conv = mpi.UnwrapMemrefOp(send.buffer)
                send.operands = [*conv.results, *send.operands[1:]]
                rewriter.replace_matched_op([conv] + self.lower_send(send), [])
            case mpi.RecvOp() as recv:
                assert isinstance(recv, mpi.RecvOp)
                conv = mpi.UnwrapMemrefOp(recv.buffer)
                recv.operands = [*conv.results, *recv.operands[1:]]
                rewriter.replace_matched_op([conv] + self.lower_recv(recv), [])
            case mpi.CommRankOp():
                rewriter.replace_matched_op(*self.lower_comm_rank())

    def lower_init(self):
        return [
            nullptr := llvm.NullOp(),
            llvm.CallOp("MPI_Init", nullptr, nullptr, result_type=builtin.i32),
        ]

    def lower_finalize(self):
        return [llvm.CallOp("MPI_Finalize", result_type=builtin.i32)]

    def lower_comm_rank(self):
        comm_world_ops, comm_world = self.target.materialize_named_constant(
            "MPI_COMM_WORLD"
        )
        return [
            *comm_world_ops,
            one := arith.Constant.from_int_and_width(1, 64),
            ptr := llvm.AllocaOp(one, builtin.i32, as_untyped_ptr=True),
            llvm.CallOp("MPI_Comm_rank", comm_world, ptr, result_type=builtin.i32),
            res := llvm.LoadOp(ptr, result_type=builtin.i32),
        ], res.results

    def lower_unwrap_memref(
        self, op: mpi.UnwrapMemrefOp
    ) -> tuple[list[Operation], list[SSAValue]]:
        return [
            index := memref.ExtractAlignedPointerAsIndexOp.get(op.ref),
            i64 := arith.IndexCastOp(index, builtin.i64),
            ptr := llvm.IntToPtrOp(i64),
            size := arith.Constant.from_int_and_width(op.ref.type.element_count(), 32),
            dtype := mpi.GetDtypeOp(op.ref.type.get_element_type()),
        ], [ptr.output, size.result, dtype.result]

    def lower_send(self, op: mpi.SendOp):
        comm_world_ops, comm_world = self.target.materialize_named_constant(
            "MPI_COMM_WORLD"
        )
        return [
            *comm_world_ops,
            llvm.CallOp("MPI_Send", *op.operands, comm_world, result_type=builtin.i32),
        ]

    def lower_recv(self, op: mpi.SendOp):
        comm_world_ops, comm_world = self.target.materialize_named_constant(
            "MPI_COMM_WORLD"
        )
        status_ignore_ops, status_ignore = self.target.materialize_named_constant(
            "MPI_STATUS_IGNORE"
        )
        return [
            *comm_world_ops,
            *status_ignore_ops,
            llvm.CallOp(
                "MPI_Recv",
                *op.operands,
                comm_world,
                status_ignore,
                result_type=builtin.i32,
            ),
        ]


@dataclass
class LowerMPIPass(ModulePass):
    name = "lower-mpi"

    vendor: str = "mpich"

    targets: ClassVar[dict[str, type[MPITargetInfo]]] = {
        "mpich": MpichTargetInfo,
        "ompi": OpenMPITargetInfo,
    }

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        assert self.vendor in self.targets
        # TODO: how to get the lib info in here?
        target = LowerMPIPass.targets[self.vendor]()

        PatternRewriteWalker(
            MpiLoweringPattern(target),
            apply_recursively=True,
        ).rewrite_module(op)

        # add globals
        for thing in target.globals_to_add():
            op.body.block.add_op(thing)

        # add func defs:
        seen_funcs: set[str] = set()
        for f in op.walk():
            if isinstance(f, llvm.CallOp) and f.callee.string_value().startswith(
                "MPI_"
            ):
                if f.callee.string_value() in seen_funcs:
                    continue
                seen_funcs.add(f.callee.string_value())
                op.body.block.add_op(f.to_declaration(llvm.LinkageAttr("external")))
