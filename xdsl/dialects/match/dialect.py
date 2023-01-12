from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *
from xdsl.dialects.IRUtils.dialect import ValueType, TypeType, OperationType, AttributeType, RangeType
from xdsl.dialects.pdl.dialect import PatternType

##############################################################################
############################# Extensions to PDL ##############################
##############################################################################


@irdl_op_definition
class MatchAndReplace(Operation):
    name: str = "match.match_and_replace"
    matched_op: Annotated[Operand, OperationType]
    pattern: Annotated[Operand, PatternType]
    body = RegionDef()


@irdl_op_definition
class Pattern(Operation):
    name: str = "match.pattern"
    body = RegionDef()
    result: Annotated[OpResult, PatternType]


@irdl_op_definition
class Capture(Operation):
    name: str = "match.capture"
    input = VarOperandDef(
        AnyOf([ValueType, OperationType, AttributeType, TypeType]))


@irdl_op_definition
class AnyInRange(Operation):
    name: str = "match.any_in_range"
    range: Annotated[Operand, RangeType]


@irdl_op_definition
class Equal(Operation):
    name: str = "match.equal"
    values = VarOperandDef(
        AnyOf([ValueType, OperationType, AttributeType, TypeType]))


Match = Dialect([
    MatchAndReplace,
    Pattern,
    Capture,
    AnyInRange,
    Equal,
], [])