from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)


@dataclass(frozen=True)
class MpichLibraryInfo:
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
class LowerMPIPass(ModulePass):
    name = "lower-mpi"

    library: str = "mpich"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        assert self.library == "mpich"
        # TODO: how to get the lib info in here?
        MpichLibraryInfo()
        walker1 = PatternRewriteWalker(
            GreedyRewritePatternApplier([]),
            apply_recursively=True,
        )

        walker1.rewrite_module(op)
