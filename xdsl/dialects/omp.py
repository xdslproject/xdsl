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
from xdsl.ir import Attribute, Dialect, EnumAttribute, OpaqueSyntaxAttribute, StrEnum
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_operand_def,
    opt_prop_def,
    region_def,
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
class ScheduleKindAttr(EnumAttribute[ScheduleKind], OpaqueSyntaxAttribute):
    name = "omp.schedulekind"
    pass


@irdl_attr_definition
class ScheduleModifierAttr(EnumAttribute[ScheduleModifier], OpaqueSyntaxAttribute):
    name = "omp.sched_mod"
    pass


@irdl_attr_definition
class OrderKindAttr(EnumAttribute[OrderKind], OpaqueSyntaxAttribute):
    name = "omp.orderkind"
    pass


@irdl_op_definition
class WsLoopOp(IRDLOperation):
    name = "omp.wsloop"

    lowerBound = var_operand_def(IntegerType | IndexType)
    upperBound = var_operand_def(IntegerType | IndexType)
    step = var_operand_def(IntegerType | IndexType)
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

    body = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "omp.yield"

    traits = frozenset([IsTerminator()])


OMP = Dialect(
    "omp", [WsLoopOp, YieldOp], [ScheduleKindAttr, ScheduleModifierAttr, OrderKindAttr]
)
