from enum import IntFlag, auto

from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
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
    Operation,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    SameVariadicOperandSize,
    base,
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
from xdsl.traits import (
    IsolatedFromAbove,
    IsTerminator,
    NoMemoryEffect,
    NoTerminator,
    OpTrait,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


class ScheduleKind(StrEnum):
    static = auto()
    dynamic = auto()
    auto = auto()


class ScheduleModifier(StrEnum):
    none = auto()
    monotonic = auto()
    nonmonotonic = auto()
    simd = auto()


class OrderKind(StrEnum):
    concurrent = auto()


class VariableCaptureKind(StrEnum):
    capture_this = auto()
    capture_by_ref = auto()
    capture_by_copy = auto()
    capture_vla_type = auto()


class OpenMPOffloadMappingFlags(IntFlag):
    """
    Copied from https://github.com/llvm/llvm-project/blob/daa2a587cc01c5656deecda7f768fed0afc1e515/llvm/include/llvm/Frontend/OpenMP/OMPConstants.h#L198
    To be used as `map_type` value of `omp.map.info`
    """

    # No flags
    NONE = 0x0
    # Allocate memory on the device and move data from host to device.
    TO = 0x01
    # Allocate memory on the device and move data from device to host.
    FROM = 0x02
    # Always perform the requested mapping action on the element, even
    # if it was already mapped before.
    ALWAYS = 0x04
    # Delete the element from the device environment, ignoring the
    # current reference count associated with the element.
    DELETE = 0x08
    # The element being mapped is a pointer-pointee pair; both the
    # pointer and the pointee should be mapped.
    PTR_AND_OBJ = 0x10
    # This flags signals that the base address of an entry should be
    # passed to the target kernel as an argument.
    TARGET_PARAM = 0x20
    # Signal that the runtime library has to return the device pointer
    # in the current position for the data being mapped. Used when we have the
    # use_device_ptr or use_device_addr clause.
    RETURN_PARAM = 0x40
    # This flag signals that the reference being passed is a pointer to
    # private data.
    PRIVATE = 0x80
    # Pass the element to the device by value.
    LITERAL = 0x100
    # Implicit map
    IMPLICIT = 0x200
    # Close is a hint to the runtime to allocate memory close to
    # the target device.
    CLOSE = 0x400
    # 0x800 is reserved for compatibility with XLC.
    # Produce a runtime error if the data is not already allocated.
    PRESENT = 0x1000
    # Increment and decrement a separate reference counter so that the data
    # cannot be unmapped within the associated region.  Thus, this flag is
    # intended to be used on 'target' and 'target data' directives because they
    # are inherently structured.  It is not intended to be used on 'target
    # enter data' and 'target exit data' directives because they are inherently
    # dynamic.
    # This is an OpenMP extension for the sake of OpenACC support.
    OMPX_HOLD = 0x2000
    # Signal that the runtime library should use args as an array of
    # descriptor_dim pointers and use args_size as dims. Used when we have
    # non-contiguous list items in target update directive
    NON_CONTIG = 0x100000000000
    # The 16 MSBs of the flags indicate whether the entry is member of some
    # struct/class.
    MEMBER_OF = 0xFFFF000000000000


class LoopWrapper(OpTrait):
    """
    Check that the omp operation is a loop wrapper as defined upstream.

    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/#loop-associated-directives
    """

    def verify(self, op: Operation) -> None:
        if not op.has_trait(NoTerminator()):
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: does not have NoTerminator trait"
            )
        if (num_regions := len(op.regions)) != 1:
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: has {num_regions} region, expected 1"
            )
        if (num_ops := len(op.regions[0].block.ops)) != 1:
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: has {num_ops} ops, expected 1"
            )
        assert (inner := op.regions[0].block.first_op), "Expected exactly 1 op"
        if not (inner.has_trait(LoopWrapper()) or isa(inner, LoopNestOp)):
            raise VerifyException(
                f"{op.name} is not a LoopWrapper: "
                f"should have a single operation which is either another LoopWrapper or {LoopNestOp.name}"
            )


@irdl_attr_definition
class ScheduleKindAttr(EnumAttribute[ScheduleKind], SpacedOpaqueSyntaxAttribute):
    name = "omp.schedulekind"


@irdl_attr_definition
class ScheduleModifierAttr(
    EnumAttribute[ScheduleModifier], SpacedOpaqueSyntaxAttribute
):
    name = "omp.sched_mod"


@irdl_attr_definition
class OrderKindAttr(EnumAttribute[OrderKind], SpacedOpaqueSyntaxAttribute):
    name = "omp.orderkind"


@irdl_attr_definition
class VariableCaptureKindAttr(
    EnumAttribute[VariableCaptureKind], SpacedOpaqueSyntaxAttribute
):
    name = "omp.variable_capture_kind"


@irdl_attr_definition
class MapBoundsType(ParametrizedAttribute, TypeAttribute):
    name = "omp.map_bounds_ty"


@irdl_op_definition
class LoopNestOp(IRDLOperation):
    name = "omp.loop_nest"

    lowerBound = var_operand_def(base(IntegerType) | base(IndexType))
    upperBound = var_operand_def(base(IntegerType) | base(IndexType))
    step = var_operand_def(base(IntegerType) | base(IndexType))

    body = region_def("single_block")

    irdl_options = [SameVariadicOperandSize()]


@irdl_op_definition
class WsLoopOp(IRDLOperation):
    name = "omp.wsloop"

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    linear_vars = var_operand_def()
    linear_step_vars = var_operand_def(i32)
    private_vars = var_operand_def()
    # TODO: this is constrained to OpenMP_PointerLikeTypeInterface upstream
    # Relatively shallow interface with just `getElementType`
    reduction_vars = var_operand_def()
    schedule_chunk = opt_operand_def()

    reductions = opt_prop_def(ArrayAttr[SymbolRefAttr])
    schedule_kind = opt_prop_def(ScheduleKindAttr)
    schedule_mod = opt_prop_def(ScheduleModifierAttr)
    simd_modifier = opt_prop_def(UnitAttr)
    nowait = opt_prop_def(UnitAttr)
    ordered = opt_prop_def(IntegerAttr[IntegerType])
    order = opt_prop_def(OrderKindAttr)
    inclusive = opt_prop_def(UnitAttr)

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(NoTerminator(), LoopWrapper())


class ProcBindKindEnum(StrEnum):
    Primary = auto()
    Master = auto()
    Close = auto()
    Spread = auto()


class ProcBindKindAttr(EnumAttribute[ProcBindKindEnum], SpacedOpaqueSyntaxAttribute):
    name = "omp.procbindkind"


@irdl_op_definition
class ParallelOp(IRDLOperation):
    name = "omp.parallel"

    allocate_vars = var_operand_def()
    allocators_vars = var_operand_def()
    if_expr = opt_operand_def(IntegerType(1))
    num_threads = opt_operand_def(base(IntegerType) | base(IndexType))
    # TODO: this is constrained to OpenMP_PointerLikeTypeInterface upstream
    # Relatively shallow interface with just `getElementType`
    private_vars = var_operand_def()
    reduction_vars = var_operand_def()

    region = region_def()

    reductions = opt_prop_def(ArrayAttr[SymbolRefAttr])
    proc_bind_kind = opt_prop_def(ProcBindKindAttr)
    privatizers = opt_prop_def(ArrayAttr[SymbolRefAttr])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "omp.yield"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class TerminatorOp(IRDLOperation):
    name = "omp.terminator"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class TargetOp(IRDLOperation):
    """
    Implementation of upstream omp.target
    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget-omptargetop
    """

    name = "omp.target"

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    depend_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    device = opt_operand_def(IntegerType | IndexType)
    has_device_addr_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    host_eval_vars = var_operand_def()
    if_expr = opt_operand_def(i1)
    in_reduction_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    is_device_ptr_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    map_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    private_vars = var_operand_def()
    thread_limit = opt_operand_def(IntegerType | IndexType)

    in_reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    nowait = opt_prop_def(UnitAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])

    region = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(IsolatedFromAbove())


@irdl_op_definition
class MapBoundsOp(IRDLOperation):
    """
    Implementation of upstream omp.map.bounds
    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompmapbounds-ompmapboundsop
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
    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompmapinfo-ompmapinfoop
    """

    name = "omp.map.info"

    var_ptr = operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    var_ptr_ptr = opt_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    members = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    bounds = var_operand_def(MapBoundsType)

    var_type = prop_def(TypeAttribute)
    members_index = opt_prop_def(ArrayAttr[i64])
    map_type = opt_prop_def(IntegerAttr)
    map_capture_type = opt_prop_def(VariableCaptureKindAttr)
    var_name = opt_prop_def(StringAttr, prop_name="name")
    partial_map = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    omp_ptr = result_def()  # TODO: OpenMP_PointerLikeTypeInterface

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def verify_(self) -> None:
        for mem in self.members:
            if not isinstance(mem.owner, MapInfoOp):
                raise VerifyException(
                    f"All `members` must be results of another {self.name}"
                )
        return super().verify_()


@irdl_op_definition
class SIMDOp(IRDLOperation):
    """
    Implementation of upstream omp.simd
    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompsimd-ompsimdop
    """

    name = "omp.simd"

    aligned_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    if_expr = opt_operand_def(i1)
    linear_vars = var_operand_def()
    linear_step_vars = var_operand_def(i32)
    nontemporal_vars = opt_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    private_vars = opt_operand_def()
    reduction_vars = opt_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface

    alignments = opt_prop_def(ArrayAttr[IntegerAttr[i64]])
    order = opt_prop_def(OrderKindAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    simdlen = opt_prop_def(IntegerAttr[i64])
    safelen = opt_prop_def(IntegerAttr[i64])

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(NoTerminator(), RecursiveMemoryEffect(), LoopWrapper())

    def verify_(self) -> None:
        if len(self.alignments.data if self.alignments else ()) != len(
            self.aligned_vars
        ):
            raise VerifyException(
                "Number of `alignments` must match number of `aligned_vars`"
            )
        if len(self.linear_vars) != len(self.linear_step_vars):
            raise VerifyException(
                "Number of `linear_vars` must match number of `linear_step_vars`"
            )
        return super().verify_()


@irdl_op_definition
class TeamsOp(IRDLOperation):
    """
    Implementation of upstream omp.teams
    https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#ompteams-ompteamsop
    """

    name = "omp.teams"

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    if_expr = opt_operand_def(i1)
    num_teams_lower = opt_operand_def(IntegerType | IndexType)
    num_teams_upper = opt_operand_def(IntegerType | IndexType)
    private_vars = var_operand_def()
    reduction_vars = var_operand_def()  # TODO: OpenMP_PointerLikeTypeInterface
    thread_limit = opt_operand_def(IntegerType | IndexType)

    reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])

    body = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(RecursiveMemoryEffect())


@irdl_op_definition
class DistributeOp(IRDLOperation):
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
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    body = region_def("single_block")

    traits = traits_def(NoTerminator(), LoopWrapper(), RecursiveMemoryEffect())


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
        SIMDOp,
        TeamsOp,
        DistributeOp,
    ],
    [
        MapBoundsType,
        OrderKindAttr,
        ProcBindKindAttr,
        ScheduleKindAttr,
        ScheduleModifierAttr,
        VariableCaptureKindAttr,
    ],
)
