from enum import auto

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    SymbolRefAttr,
    UnitAttr,
    i32,
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
    AttrSizedOperandSegments,
    IRDLOperation,
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
from xdsl.traits import IsTerminator


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

    linear_vars = var_operand_def()
    linear_step_vars = var_operand_def(i32)
    # TODO: this is constrained to OpenMP_PointerLikeTypeInterface upstream
    # Relatively shallow interface with just `getElementType`
    reduction_vars = var_operand_def()
    schedule_chunk_var = opt_operand_def()

    reductions = opt_prop_def(ArrayAttr[SymbolRefAttr])
    schedule_val = opt_prop_def(ScheduleKindAttr)
    schedule_modifier = opt_prop_def(ScheduleModifierAttr)
    simd_modifier = opt_prop_def(UnitAttr)
    nowait = opt_prop_def(UnitAttr)
    ordered_val = opt_prop_def(IntegerAttr[IntegerType])
    order_val = opt_prop_def(OrderKindAttr)
    inclusive = opt_prop_def(UnitAttr)

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


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

    if_expr_var = opt_operand_def(IntegerType(1))
    num_threads_var = opt_operand_def(base(IntegerType) | base(IndexType))
    allocate_vars = var_operand_def()
    allocators_vars = var_operand_def()
    # TODO: this is constrained to OpenMP_PointerLikeTypeInterface upstream
    # Relatively shallow interface with just `getElementType`
    reduction_vars = var_operand_def()
    private_vars = var_operand_def()

    region = region_def()

    reductions = opt_prop_def(ArrayAttr[SymbolRefAttr])
    proc_bind_val = opt_prop_def(ProcBindKindAttr)
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


OMP = Dialect(
    "omp",
    [
        ParallelOp,
        TerminatorOp,
        WsLoopOp,
        LoopNestOp,
        YieldOp,
    ],
    [
        OrderKindAttr,
        ProcBindKindAttr,
        ScheduleKindAttr,
        ScheduleModifierAttr,
    ],
)
