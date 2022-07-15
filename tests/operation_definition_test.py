from __future__ import annotations

from xdsl.ir import Operation
from xdsl.irdl import (irdl_op_definition, OperandDef, ResultDef, AttributeDef,
                       RegionDef, AnyAttr, OpDef)

#  ___ ____  ____  _     ____        __
# |_ _|  _ \|  _ \| |   |  _ \  ___ / _|
#  | || |_) | | | | |   | | | |/ _ \ |_
#  | ||  _ <| |_| | |___| |_| |  __/  _|
# |___|_| \_\____/|_____|____/ \___|_|
#


@irdl_op_definition
class OpDefTestOp(Operation):
    name = "test.op_def_test"

    operand = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
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
