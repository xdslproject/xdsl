from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntFlag, auto
from typing import ClassVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    i1,
    i32,
    i64,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AtLeast,
    AttrSizedOperandSegments,
    IntVarConstraint,
    IRDLOperation,
    Operand,
    Operation,
    RangeOf,
    SameVariadicOperandSize,
    VarOperand,
    base,
    eq,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    NoMemoryEffect,
    NoTerminator,
    Pure,
    RecursiveMemoryEffect,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException


class OpenMPOffloadMappingFlags(IntFlag):
    """
    Copied from [OMPConstants.h](https://github.com/llvm/llvm-project/blob/daa2a587cc01c5656deecda7f768fed0afc1e515/llvm/include/llvm/Frontend/OpenMP/OMPConstants.h#L198)


    `map_type` attribute of `omp.map.info` is a 64 bit integer, with each bit specifying the flags described below.
    Flags can be combined with `|` (bitwise or) operator.
    """

    NONE = 0x0
    """ No flags """
    TO = 0x01
    """ Allocate memory on the device and move data from host to device. """
    FROM = 0x02
    """ Allocate memory on the device and move data from device to host. """
    ALWAYS = 0x04
    """ Always perform the requested mapping action on the element, even if it was already mapped before. """
    DELETE = 0x08
    """ Delete the element from the device environment, ignoring the current reference count associated with the element. """
    PTR_AND_OBJ = 0x10
    """ The element being mapped is a pointer-pointee pair; both the pointer and the pointee should be mapped. """
    TARGET_PARAM = 0x20
    """ This flags signals that the base address of an entry should be passed to the target kernel as an argument. """
    RETURN_PARAM = 0x40
    """
    Signal that the runtime library has to return the device pointer in the current position for the data being mapped.
    Used when we have the use_device_ptr or use_device_addr clause.
    """
    PRIVATE = 0x80
    """ This flag signals that the reference being passed is a pointer to private data. """
    LITERAL = 0x100
    """ Pass the element to the device by value. """
    IMPLICIT = 0x200
    """ Implicit map """
    CLOSE = 0x400
    """ Close is a hint to the runtime to allocate memory close to the target device. """

    # 0x800 is reserved for compatibility with XLC.

    PRESENT = 0x1000
    """ Produce a runtime error if the data is not already allocated. """
    OMPX_HOLD = 0x2000
    """
    Increment and decrement a separate reference counter so that the data cannot be unmapped within the associated region.
    Thus, this flag is intended to be used on 'target' and 'target data' directives because they are inherently structured.
    It is not intended to be used on 'target enter data' and 'target exit data' directives because they are inherently dynamic.
    This is an OpenMP extension for the sake of OpenACC support.
    """
    NON_CONTIG = 0x100000000000
    """
    Signal that the runtime library should use args as an array of descriptor_dim pointers and use args_size as dims.
    Used when we have non-contiguous list items in target update directive
    """
    MEMBER_OF = 0xFFFF000000000000
    """ The 16 MSBs of the flags indicate whether the entry is member of some struct/class. """

    def __iter__(self):
        """
        available in the standard library since python 3.11
        """
        return (flag for flag in type(self) if self & flag)


def verify_map_vars(
    vars: VarOperand,
    op_name: str,
    *,
    disallowed_types: OpenMPOffloadMappingFlags = OpenMPOffloadMappingFlags.NONE,
):
    for var in vars:
        if not isinstance(owner := var.owner, MapInfoOp):
            raise VerifyException(
                f"All mapped operands of {op_name} must be results of a {MapInfoOp.name}"
            )
        for t in disallowed_types:
            if owner.map_type.value.data & t:
                raise VerifyException(f"Cannot have map_type {t.name} in {op_name}")


class ScheduleKind(StrEnum):
    STATIC = auto()
    DYNAMIC = auto()
    AUTO = auto()


class ScheduleModifier(StrEnum):
    NONE = auto()
    MONOTONIC = auto()
    NONMONOTONIC = auto()
    SIMD = auto()


class OrderKind(StrEnum):
    CONCURRENT = auto()


class OrderModifier(StrEnum):
    REPRODUCIBLE = auto()
    UNCONSTRAINED = auto()


class DependKind(StrEnum):
    TASKDEPENDIN = auto()
    TASKDEPENDOUT = auto()
    TASKDEPENDINOUT = auto()
    TASKDEPENDMUTEXINOUTSET = auto()
    TASKDEPENDINOUTSET = auto()


class VariableCaptureKind(StrEnum):
    THIS = "This"
    BY_REF = "ByRef"
    BY_COPY = "ByCopy"
    VLA_TYPE = "VLAType"


class DataSharingClauseKind(StrEnum):
    PRIVATE = auto()
    FIRSTPRIVATE = auto()


class ClauseRequiresKind(StrEnum):
    NONE = auto()
    REVERSE_OFFLOAD = auto()
    UNIFIED_ADDRESS = auto()
    UNIFIED_SHARED_MEMORY = auto()
    DYNAMIC_ALLOCATORS = auto()


class DeclareTargetDeviceTypeKind(StrEnum):
    ANY = auto()
    HOST = auto()
    NOHOST = auto()


class DeclareTargetCaptureClauseKind(StrEnum):
    TO = auto()
    LINK = auto()
    ENTER = auto()


class ReductionModifier(StrEnum):
    DEFAULTMOD = auto()
    INSCAN = auto()
    TASK = auto()


class LoopWrapper(NoTerminator):
    """
    Check that the omp operation is a loop wrapper as defined upstream.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/#loop-associated-directives)
    """

    def verify(self, op: Operation) -> None:
        if (num_regions := len(op.regions)) != 1:
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: has {num_regions} region, expected 1"
            )
        if (num_blocks := len(op.regions[0].blocks)) != 1:
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: has {num_blocks} blocks, expected 1"
            )
        if (num_ops := len(op.regions[0].block.ops)) != 1:
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: has {num_ops} ops, expected 1"
            )
        assert (inner := op.regions[0].block.first_op) is not None
        if not (inner.has_trait(LoopWrapper()) or isinstance(inner, LoopNestOp)):
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: "
                f"should have a single operation which is either another LoopWrapper or {LoopNestOp.name}"
            )
        return super().verify(op)


class BlockArgOpenMPOperation(IRDLOperation, ABC):
    """
    Verifies that the operation has the appropriate number of block arguments corresponding to the following operands:
    `host_eval_vars`, `in_reduction_vars`, `map_vars`, `private_vars`, `reduction_vars`,
    `task_reduction_vars`, `use_device_addr_vars` and `use_device_ptr_vars`.
    """

    def verify_(self) -> None:
        if len(self.regions) < 1:
            raise VerifyException(f"Expected {self.name} to have at least 1 region")
        if len(self.regions[0].blocks) < 1:
            raise VerifyException(
                f"Expected {self.name} to have at least 1 block in its first region"
            )
        expected = self.num_block_args()

        if (actual := len(self.regions[0].blocks[0].args)) < expected:
            raise VerifyException(
                f"{self.name} expected to have at least {expected} block argument(s), got {actual}"
            )

    @abstractmethod
    def num_block_args(self) -> int: ...


_ui64 = IntegerType(64, Signedness.UNSIGNED)


@irdl_attr_definition
class ScheduleKindAttr(EnumAttribute[ScheduleKind], SpacedOpaqueSyntaxAttribute):
    name = "omp.schedulekind"


@irdl_attr_definition
class ScheduleModifierAttr(
    EnumAttribute[ScheduleModifier], SpacedOpaqueSyntaxAttribute
):
    name = "omp.sched_mod"


@irdl_attr_definition
class DependKindAttr(EnumAttribute[DependKind], SpacedOpaqueSyntaxAttribute):
    name = "omp.clause_task_depend"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_string(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DependKind:
        with parser.in_parens():
            return parser.parse_str_enum(DependKind)


@irdl_attr_definition
class OrderKindAttr(EnumAttribute[OrderKind], SpacedOpaqueSyntaxAttribute):
    name = "omp.orderkind"


@irdl_attr_definition
class ReductionModifierAttr(
    EnumAttribute[ReductionModifier], SpacedOpaqueSyntaxAttribute
):
    name = "omp.reduction_modifier"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_string(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> ReductionModifier:
        with parser.in_parens():
            return parser.parse_str_enum(ReductionModifier)


@irdl_attr_definition
class OrderModifierAttr(EnumAttribute[OrderModifier], SpacedOpaqueSyntaxAttribute):
    name = "omp.order_mod"


@irdl_attr_definition
class VariableCaptureKindAttr(
    EnumAttribute[VariableCaptureKind], SpacedOpaqueSyntaxAttribute
):
    name = "omp.variable_capture_kind"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_string(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> VariableCaptureKind:
        with parser.in_parens():
            return parser.parse_str_enum(VariableCaptureKind)


@irdl_attr_definition
class DeclareTargetDeviceTypeAttr(
    EnumAttribute[DeclareTargetDeviceTypeKind], SpacedOpaqueSyntaxAttribute
):
    """
    Implementation of upstream omp.device_type
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#declaretargetdevicetypeattr).
    """

    name = "omp.device_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DeclareTargetDeviceTypeKind:
        with parser.in_parens():
            return parser.parse_str_enum(DeclareTargetDeviceTypeKind)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_string(self.data)


@irdl_attr_definition
class DeclareTargetCaptureClauseAttr(
    EnumAttribute[DeclareTargetCaptureClauseKind], SpacedOpaqueSyntaxAttribute
):
    """
    Implementation of upstream omp.capture_clause
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#declaretargetcaptureclauseattr).
    """

    name = "omp.capture_clause"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DeclareTargetCaptureClauseKind:
        with parser.in_parens():
            return parser.parse_str_enum(DeclareTargetCaptureClauseKind)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_string(self.data)


@irdl_attr_definition
class DeclareTargetAttr(ParametrizedAttribute, SpacedOpaqueSyntaxAttribute):
    """
    Implementation of upstream omp.declaretarget
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#declaretargetattr).
    """

    name = "omp.declaretarget"

    device_type: DeclareTargetDeviceTypeAttr
    capture_clause: DeclareTargetCaptureClauseAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_keyword("device_type")
            parser.parse_punctuation("=")
            device_type = DeclareTargetDeviceTypeAttr(
                DeclareTargetDeviceTypeAttr.parse_parameter(parser)
            )
            parser.parse_punctuation(",")
            parser.parse_keyword("capture_clause")
            parser.parse_punctuation("=")
            capture = DeclareTargetCaptureClauseAttr(
                DeclareTargetCaptureClauseAttr.parse_parameter(parser)
            )
            return [device_type, capture]

    def print_parameters(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string("device_type = ")
            self.device_type.print_parameter(printer)
            printer.print_string(", capture_clause = ")
            self.capture_clause.print_parameter(printer)


@irdl_attr_definition
class ClauseRequiresKindAttr(
    EnumAttribute[ClauseRequiresKind], SpacedOpaqueSyntaxAttribute
):
    """
    Implementation of upstream omp.clause_requires
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#clauserequiresattr).
    """

    name = "omp.clause_requires"


@irdl_attr_definition
class DataSharingClauseAttr(
    EnumAttribute[DataSharingClauseKind], SpacedOpaqueSyntaxAttribute
):
    """
    Implementation of upstream omp.data_sharing_type
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#datasharingclausetypeattr).
    """

    name = "omp.data_sharing_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DataSharingClauseKind:
        with parser.in_braces():
            parser.parse_keyword("type")
            parser.parse_punctuation("=")
            return parser.parse_str_enum(DataSharingClauseKind)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_braces():
            printer.print_string("type = ")
            printer.print_string(self.data)


@irdl_attr_definition
class VersionAttr(ParametrizedAttribute, SpacedOpaqueSyntaxAttribute):
    """
    Implementation of upstream omp.version
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#versionattr).
    """

    name = "omp.version"

    version: IntAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_keyword("version")
            parser.parse_punctuation("=")
            version = parser.parse_integer(allow_boolean=False)
            return [IntAttr(version)]

    def print_parameters(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string("version = ")
            printer.print_int(self.version.data)


@irdl_attr_definition
class MapBoundsType(ParametrizedAttribute, TypeAttribute):
    name = "omp.map_bounds_ty"


@irdl_op_definition
class LoopNestOp(IRDLOperation):
    name = "omp.loop_nest"

    lowerBound = var_operand_def(base(IntegerType) | base(IndexType))
    upperBound = var_operand_def(base(IntegerType) | base(IndexType))
    step = var_operand_def(base(IntegerType) | base(IndexType))

    loop_inclusive = opt_prop_def(UnitAttr)

    body = region_def("single_block")

    irdl_options = [SameVariadicOperandSize()]

    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class WsLoopOp(BlockArgOpenMPOperation):
    name = "omp.wsloop"

    LINEAR_COUNT: ClassVar = IntVarConstraint("LINEAR_COUNT", AnyInt())
    REDUCTION_COUNT: ClassVar = IntVarConstraint("REDUCTION_COUNT", AnyInt())

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    linear_vars = var_operand_def(RangeOf(AnyAttr()).of_length(LINEAR_COUNT))
    linear_step_vars = var_operand_def(RangeOf(eq(i32)).of_length(LINEAR_COUNT))
    private_vars = var_operand_def()
    # TODO: this is constrained to OpenMP_PointerLikeTypeInterface upstream
    # Relatively shallow interface with just `getElementType`
    reduction_vars = var_operand_def(RangeOf(AnyAttr()).of_length(REDUCTION_COUNT))
    schedule_chunk = opt_operand_def()

    reduction_syms = opt_prop_def(
        ArrayAttr.constr(RangeOf(base(SymbolRefAttr)).of_length(REDUCTION_COUNT))
    )
    reduction_mod = opt_prop_def(ReductionModifierAttr)
    reduction_byref = opt_prop_def(DenseArrayBase[i1])
    schedule_kind = opt_prop_def(ScheduleKindAttr)
    schedule_mod = opt_prop_def(ScheduleModifierAttr)
    schedule_simd = opt_prop_def(UnitAttr)
    nowait = opt_prop_def(UnitAttr)
    ordered = opt_prop_def(IntegerAttr.constr(value=AtLeast(0), type=base(IntegerType)))
    order = opt_prop_def(OrderKindAttr)
    order_mod = opt_prop_def(OrderModifierAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    private_needs_barrier = opt_prop_def(UnitAttr)

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(LoopWrapper(), RecursiveMemoryEffect())

    def num_block_args(self) -> int:
        return len(self.private_vars) + len(self.reduction_vars)


class ProcBindKindEnum(StrEnum):
    PRIMARY = auto()
    MASTER = auto()
    CLOSE = auto()
    SPREAD = auto()


class ProcBindKindAttr(EnumAttribute[ProcBindKindEnum], SpacedOpaqueSyntaxAttribute):
    name = "omp.procbindkind"


@irdl_op_definition
class ParallelOp(BlockArgOpenMPOperation):
    name = "omp.parallel"

    allocate_vars = var_operand_def()
    allocators_vars = var_operand_def()
    if_expr = opt_operand_def(i1)
    num_threads = opt_operand_def(base(IntegerType) | base(IndexType))
    # TODO: this is constrained to OpenMP_PointerLikeTypeInterface upstream
    # Relatively shallow interface with just `getElementType`
    private_vars = var_operand_def()
    reduction_vars = var_operand_def()

    region = region_def()

    reductions = opt_prop_def(ArrayAttr[SymbolRefAttr])
    proc_bind_kind = opt_prop_def(ProcBindKindAttr)
    privatizers = opt_prop_def(ArrayAttr[SymbolRefAttr])
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    reduction_mod = opt_prop_def(ReductionModifierAttr)
    reduction_byref = opt_prop_def(DenseArrayBase[i1])
    reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(RecursiveMemoryEffect())

    def num_block_args(self) -> int:
        return len(self.private_vars) + len(self.reduction_vars)


@irdl_op_definition
class DeclareReductionOp(IRDLOperation):
    """
    Implementation of upstream omp.declare_reduction
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompdeclare_reduction-ompdeclarereductionop).
    """

    name = "omp.declare_reduction"

    sym_name = prop_def(SymbolNameConstraint())
    var_type = prop_def(TypeAttribute, prop_name="type")

    alloc_region = region_def()
    init_region = region_def()
    reduction_region = region_def()
    atomic_reduction_region = region_def()
    cleanup_region = region_def()

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())

    assembly_format = """
        $sym_name `:` $type attr-dict-with-keyword
        ( `alloc` $alloc_region^ )?
        `init` $init_region
        `combiner` $reduction_region
        ( `atomic` $atomic_reduction_region^ )?
        ( `cleanup` $cleanup_region^ )?
    """

    def verify_(self) -> None:
        if len(self.alloc_region.blocks) > 1:
            raise VerifyException(
                f"{self.name} should have at most 1 block in alloc_region"
            )


@irdl_op_definition
class PrivateClauseOp(IRDLOperation):
    """
    Implementation of upstream omp.private
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompprivate-ompprivateclauseop).
    """

    name = "omp.private"

    sym_name = prop_def(SymbolNameConstraint())
    var_type = prop_def(TypeAttribute, prop_name="type")
    data_sharing_type = prop_def(DataSharingClauseAttr)

    init_region = region_def()
    copy_region = region_def()
    dealloc_region = region_def()

    traits = traits_def(IsolatedFromAbove())

    assembly_format = """
        $data_sharing_type $sym_name `:` $type
        (`init` $init_region^)?
        (`copy` $copy_region^)?
        (`dealloc` $dealloc_region^)?
        attr-dict
    """


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "omp.yield"

    assembly_format = "( `(` $arguments^ `:` type($arguments) `)` )? attr-dict"

    traits = traits_def(
        IsTerminator(),
        Pure(),
        HasParent(
            LoopNestOp,
            # TODO: add these when they are implemented
            # AtomicUpdateOp,
            PrivateClauseOp,
            DeclareReductionOp,
        ),
    )


@irdl_op_definition
class TerminatorOp(IRDLOperation):
    name = "omp.terminator"

    traits = traits_def(IsTerminator(), Pure())


@irdl_op_definition
class TargetOp(BlockArgOpenMPOperation):
    """
    Implementation of upstream omp.target
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget-omptargetop).
    """

    name = "omp.target"

    DEP_COUNT: ClassVar = IntVarConstraint("DEP_COUNT", AnyInt())

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    depend_vars = var_operand_def(
        RangeOf(
            AnyAttr(),  # TODO: OpenMP_PointerLikeTypeInterface
        ).of_length(DEP_COUNT)
    )
    device = opt_operand_def(IntegerType)
    has_device_addr_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    host_eval_vars = var_operand_def()
    if_expr = opt_operand_def(i1)
    in_reduction_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    is_device_ptr_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    map_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    private_vars = var_operand_def()
    thread_limit = opt_operand_def(IntegerType | IndexType)

    bare = opt_prop_def(UnitAttr)
    depend_kinds = opt_prop_def(
        ArrayAttr.constr(RangeOf(base(DependKindAttr)).of_length(DEP_COUNT))
    )
    in_reduction_byref = opt_prop_def(DenseArrayBase[i1])
    in_reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    nowait = opt_prop_def(UnitAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    private_needs_barrier = opt_prop_def(UnitAttr)
    private_maps = opt_prop_def(DenseArrayBase[i64])

    region = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(IsolatedFromAbove())

    def num_block_args(self) -> int:
        return (
            len(self.host_eval_vars)
            + len(self.in_reduction_vars)
            + len(self.map_vars)
            + len(self.private_vars)
        )

    def verify_(self) -> None:
        verify_map_vars(
            self.map_vars,
            self.name,
            disallowed_types=OpenMPOffloadMappingFlags.DELETE,
        )
        return super().verify_()


@irdl_op_definition
class MapBoundsOp(IRDLOperation):
    """
    Implementation of upstream omp.map.bounds
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompmapbounds-ompmapboundsop).
    """

    name = "omp.map.bounds"

    lower_bound = opt_operand_def(IntegerType | IndexType)
    upper_bound = opt_operand_def(IntegerType | IndexType)
    extent = opt_operand_def(IntegerType | IndexType)
    stride = opt_operand_def(IntegerType | IndexType)
    start_idx = opt_operand_def(IntegerType | IndexType)

    stride_in_bytes = prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    res = result_def(MapBoundsType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class MapInfoOp(IRDLOperation):
    """
    Implementation of upstream omp.map.info
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompmapinfo-ompmapinfoop).
    """

    name = "omp.map.info"

    var_ptr = operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    var_ptr_ptr = opt_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    members = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    bounds = var_operand_def(MapBoundsType)

    var_type = prop_def(TypeAttribute)
    map_type = prop_def(IntegerAttr[_ui64])
    """
    To set or test flags in `map_type` use the bits defined in `OpenMPOffloadMappingFlags`
    """
    map_capture_type = prop_def(VariableCaptureKindAttr)
    members_index = opt_prop_def(ArrayAttr[ArrayAttr[IntegerAttr[i64]]])
    var_name = opt_prop_def(StringAttr, prop_name="name")
    partial_map = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    omp_ptr = result_def()  # TODO: OpenMP_PointerLikeTypeInterface

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def verify_(self) -> None:
        verify_map_vars(self.members, self.name)
        return super().verify_()


@irdl_op_definition
class SimdOp(BlockArgOpenMPOperation):
    """
    Implementation of upstream omp.simd
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompsimd-ompsimdop).
    """

    name = "omp.simd"

    ALIGN_COUNT: ClassVar = IntVarConstraint("ALIGN_COUNT", AnyInt())
    LINEAR_COUNT: ClassVar = IntVarConstraint("LINEAR_COUNT", AnyInt())

    aligned_vars = var_operand_def(
        RangeOf(
            AnyAttr(),  # TODO: OpenMP_PointerLikeTypeInterface
        ).of_length(ALIGN_COUNT)
    )
    if_expr = opt_operand_def(i1)
    linear_vars = var_operand_def(RangeOf(AnyAttr()).of_length(LINEAR_COUNT))
    linear_step_vars = var_operand_def(RangeOf(eq(i32)).of_length(LINEAR_COUNT))
    nontemporal_vars = opt_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    private_vars = var_operand_def()
    reduction_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface

    alignments = opt_prop_def(
        ArrayAttr.constr(RangeOf(base(IntegerAttr[i64])).of_length(ALIGN_COUNT))
    )
    order = opt_prop_def(OrderKindAttr)
    order_mod = opt_prop_def(OrderModifierAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    reduction_mod = opt_prop_def(ReductionModifierAttr)
    reduction_byref = opt_prop_def(DenseArrayBase[i1])
    reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    simdlen = opt_prop_def(IntegerAttr.constr(value=AtLeast(1), type=eq(i64)))
    safelen = opt_prop_def(IntegerAttr.constr(value=AtLeast(1), type=eq(i64)))

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(RecursiveMemoryEffect(), LoopWrapper())

    def num_block_args(self) -> int:
        return len(self.private_vars) + len(self.reduction_vars)

    def verify_(self) -> None:
        if self.simdlen and self.safelen:
            if self.simdlen.value.data > self.safelen.value.data:
                raise VerifyException(
                    f"in {self.name} `safelen` must be greater than or equal to `simdlen`"
                )
        return super().verify_()


@irdl_op_definition
class TeamsOp(BlockArgOpenMPOperation):
    """
    Implementation of upstream omp.teams
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompteams-ompteamsop).
    """

    name = "omp.teams"

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    if_expr = opt_operand_def(i1)
    num_teams_lower = opt_operand_def(IntegerType)
    num_teams_upper = opt_operand_def(IntegerType)
    private_vars = var_operand_def()
    reduction_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    thread_limit = opt_operand_def(IntegerType)

    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    reduction_mod = opt_prop_def(ReductionModifierAttr)
    reduction_byref = opt_prop_def(DenseArrayBase[i1])
    reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])

    body = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(RecursiveMemoryEffect())

    def num_block_args(self) -> int:
        return len(self.private_vars) + len(self.reduction_vars)


@irdl_op_definition
class DistributeOp(BlockArgOpenMPOperation):
    """
    Implementation of upstream omp.distribute
    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompdistribute-ompdistributeop
    """

    name = "omp.distribute"

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    dist_schedule_chunk_size = opt_operand_def(IntegerType | IndexType)
    private_vars = var_operand_def()

    dist_schedule_static = opt_prop_def(UnitAttr)
    order = opt_prop_def(OrderKindAttr)
    order_mod = opt_prop_def(OrderModifierAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    body = region_def("single_block")

    traits = traits_def(LoopWrapper(), RecursiveMemoryEffect())

    def num_block_args(self) -> int:
        return len(self.private_vars)

    def verify_(self) -> None:
        if bool(self.dist_schedule_chunk_size) != bool(self.dist_schedule_static):
            raise VerifyException(
                f"{self.name} should have either both dist_schedule_static and dist_schedule_chunk_size, or neither."
            )
        return super().verify_()


class TargetTaskBasedDataOp(IRDLOperation):
    """
    Base class representing target data movement operations which are task based,
    this includes enter and exit data pragmas along with data update
    """

    DEP_COUNT: ClassVar = IntVarConstraint("DEP_COUNT", AnyInt())

    depend_vars = var_operand_def(
        RangeOf(
            AnyAttr(),  # TODO: OpenMP_PointerLikeTypeInterface
        ).of_length(DEP_COUNT)
    )
    device = opt_operand_def(IntegerType | IndexType)
    if_expr = opt_operand_def(i1)
    mapped_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface

    depend_kinds = opt_prop_def(
        ArrayAttr.constr(RangeOf(base(DependKindAttr)).of_length(DEP_COUNT))
    )
    nowait = opt_prop_def(UnitAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class TargetEnterDataOp(TargetTaskBasedDataOp):
    """
    Implementation of upstream omp.target_enter_data
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget_enter_data-omptargetenterdataop).
    """

    name = "omp.target_enter_data"

    def verify_(self) -> None:
        verify_map_vars(
            self.mapped_vars,
            self.name,
            disallowed_types=OpenMPOffloadMappingFlags.FROM
            | OpenMPOffloadMappingFlags.DELETE,
        )
        return super().verify_()


@irdl_op_definition
class TargetExitDataOp(TargetTaskBasedDataOp):
    """
    Implementation of omp.target_exit_data
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget_enter_data-omptargetenterdataop).
    """

    name = "omp.target_exit_data"

    def verify_(self) -> None:
        verify_map_vars(
            self.mapped_vars,
            self.name,
            disallowed_types=OpenMPOffloadMappingFlags.TO,
        )
        return super().verify_()


@irdl_op_definition
class TargetUpdateOp(TargetTaskBasedDataOp):
    """
    Implementation of upstream omp.target_update
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget_update-omptargetupdateop).
    """

    name = "omp.target_update"

    def verify_(self) -> None:
        verify_map_vars(
            self.mapped_vars,
            self.name,
            disallowed_types=OpenMPOffloadMappingFlags.DELETE,
        )
        mapped = defaultdict[Operand, OpenMPOffloadMappingFlags](
            lambda: OpenMPOffloadMappingFlags.NONE
        )
        one_of = OpenMPOffloadMappingFlags.TO | OpenMPOffloadMappingFlags.FROM
        for var in self.mapped_vars:
            assert isinstance(owner := var.owner, MapInfoOp)

            mapped[owner.var_ptr] |= owner.map_type.value.data
            if (mapped[owner.var_ptr] & one_of).bit_count() != 1:
                raise VerifyException(
                    f"{self.name} expected to have exactly one of TO or FROM as map_type"
                )
        return super().verify_()


@irdl_op_definition
class TargetDataOp(BlockArgOpenMPOperation):
    """
    Implementation of upstream omp.target_data
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget_data-omptargetdataop).
    """

    name = "omp.target_data"

    device = opt_operand_def(IntegerType)
    if_expr = opt_operand_def(i1)
    mapped_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    use_device_addr_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    use_device_ptr_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface

    region = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def num_block_args(self) -> int:
        # NOTE: Unlike TargetOp `mapped_vars` are not passed as block args.
        return len(self.use_device_addr_vars) + len(self.use_device_ptr_vars)

    def verify_(self) -> None:
        verify_map_vars(
            self.mapped_vars,
            self.name,
            disallowed_types=(OpenMPOffloadMappingFlags.DELETE),
        )
        return super().verify_()


OMP = Dialect(
    "omp",
    [
        ParallelOp,
        TerminatorOp,
        WsLoopOp,
        LoopNestOp,
        YieldOp,
        TargetOp,
        MapBoundsOp,
        MapInfoOp,
        SimdOp,
        TeamsOp,
        DistributeOp,
        PrivateClauseOp,
        TargetEnterDataOp,
        TargetExitDataOp,
        TargetUpdateOp,
        TargetDataOp,
        DeclareReductionOp,
    ],
    [
        ClauseRequiresKindAttr,
        DataSharingClauseAttr,
        DeclareTargetAttr,
        DeclareTargetCaptureClauseAttr,
        DeclareTargetDeviceTypeAttr,
        OrderKindAttr,
        ProcBindKindAttr,
        ScheduleKindAttr,
        ScheduleModifierAttr,
        DependKindAttr,
        MapBoundsType,
        VariableCaptureKindAttr,
        VersionAttr,
        ReductionModifierAttr,
        OrderModifierAttr,
    ],
)
