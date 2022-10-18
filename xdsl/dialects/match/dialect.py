from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.IRUtils.dialect import ValueType, TypeType, OperationType, AttributeType, RangeType
from xdsl.dialects.pdl.dialect import PatternType


# extensions to PDL
@dataclass
class Match:
    ctx: MLContext

    def __post_init__(self):
        # Ops for matching
        self.ctx.register_op(MatchAndReplace)
        self.ctx.register_op(Pattern)
        self.ctx.register_op(Capture)
        self.ctx.register_op(AnyInRange)
        self.ctx.register_op(Equal)


@irdl_op_definition
class MatchAndReplace(Operation):
    name: str = "match.match_and_replace"
    matched_op = OperandDef(OperationType)
    pattern = OperandDef(PatternType)
    body = RegionDef()


@irdl_op_definition
class Pattern(Operation):
    name: str = "match.pattern"
    body = RegionDef()
    result = ResultDef(PatternType)


@irdl_op_definition
class Capture(Operation):
    name: str = "match.capture"
    input = VarOperandDef(
        AnyOf([ValueType, OperationType, AttributeType, TypeType]))


@irdl_op_definition
class AnyInRange(Operation):
    name: str = "match.any_in_range"
    range = OperandDef(RangeType)


@irdl_op_definition
class Equal(Operation):
    name: str = "match.equal"
    values = VarOperandDef(
        AnyOf([ValueType, OperationType, AttributeType, TypeType]))
