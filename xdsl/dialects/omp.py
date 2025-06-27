from enum import auto
from typing import ClassVar, cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
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
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
)
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrSizedOperandSegments,
    IntVarConstraint,
    IRDLOperation,
    RangeOf,
    SameVariadicOperandSize,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    opt_operand_def,
    opt_prop_def,
    region_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsolatedFromAbove, IsTerminator, NoTerminator
from xdsl.utils.exceptions import VerifyException


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


class DependKind(StrEnum):
    taskdependin = auto()
    taskdependout = auto()
    taskdependinout = auto()
    taskdependmutexinoutset = auto()
    taskdependinoutset = auto()


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
        printer.print_string("(" + self.data + ")")

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DependKind:
        parser.parse_punctuation("(")
        res = parser.parse_str_enum(cls.enum_type)
        parser.parse_punctuation(")")
        return cast(DependKind, res)


@irdl_attr_definition
class OrderKindAttr(EnumAttribute[OrderKind], SpacedOpaqueSyntaxAttribute):
    name = "omp.orderkind"


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

    traits = traits_def(NoTerminator())

    def verify_(self) -> None:
        if len(self.body.blocks) == 1 and len(self.body.block.ops) != 1:
            raise VerifyException(
                f"Body of {self.name} operation body must consist of one loop nest"
            )


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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/ODS/#omptarget-omptargetop).
    """

    name = "omp.target"

    DEP_COUNT: ClassVar = IntVarConstraint("DEP_COUNT", AnyInt())

    allocate_vars = var_operand_def()
    allocator_vars = var_operand_def()
    depend_vars = var_operand_def(
        RangeOf(
            AnyAttr(),  # TODO: OpenMP_PointerLikeTypeInterface
            length=DEP_COUNT,
        )
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
        ArrayAttr.constr(RangeOf(base(DependKindAttr), length=DEP_COUNT))
    )
    in_reduction_byref = opt_prop_def(DenseIntOrFPElementsAttr[i1])
    in_reduction_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    nowait = opt_prop_def(UnitAttr)
    private_syms = opt_prop_def(ArrayAttr[SymbolRefAttr])
    private_needs_barrier = opt_prop_def(UnitAttr)
    private_maps = opt_prop_def(DenseIntOrFPElementsAttr[i64])

    region = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(IsolatedFromAbove())


OMP = Dialect(
    "omp",
    [
        ParallelOp,
        TerminatorOp,
        WsLoopOp,
        LoopNestOp,
        YieldOp,
        TargetOp,
    ],
    [
        OrderKindAttr,
        ProcBindKindAttr,
        ScheduleKindAttr,
        ScheduleModifierAttr,
        DependKindAttr,
    ],
)
