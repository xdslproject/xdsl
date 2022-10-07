from __future__ import annotations
from typing import Annotated

from xdsl.ir import OpResult, Operation, SSAValue
from xdsl.irdl import (S_OperandDef, irdl_op_definition, OperandDef, ResultDef,
                       S_ResultDef, AttributeDef, RegionDef, AnyAttr, OpDef)

#  ___ ____  ____  _     ____        __
# |_ _|  _ \|  _ \| |   |  _ \  ___ / _|
#  | || |_) | | | | |   | | | |/ _ \ |_
#  | ||  _ <| |_| | |___| |_| |  __/  _|
# |___|_| \_\____/|_____|____/ \___|_|
#


@irdl_op_definition
class OpDefTestOp(Operation):
    name = "test.op_def_test"

    operand: S_OperandDef[Annotated[SSAValue, AnyAttr]]
    result: S_ResultDef[Annotated[OpResult, AnyAttr]]
    attr = AttributeDef(AnyAttr())
    region = RegionDef()


def test_get_definition():
    """Test retrieval of an IRDL definition from an operation"""
    assert OpDefTestOp.irdl_definition == OpDef(
        "test.op_def_test",
        operands=[("operand", OperandDef(AnyAttr()))],
        results=[("result", ResultDef(AnyAttr()))],
        attributes={"attr": AttributeDef(AnyAttr())},
        regions=[("region", RegionDef())])
