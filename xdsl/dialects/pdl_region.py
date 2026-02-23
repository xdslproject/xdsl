from __future__ import annotations

from xdsl.dialects.pdl import TypeType, ValueType
from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_operand_def,
    result_def,
)


@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "pdl_region.region"
    value_type = opt_operand_def(TypeType)
    value = result_def(ValueType)

    assembly_format = "(`:` $value_type^)? attr-dict"


@irdl_attr_definition
class RegionType(ParametrizedAttribute, TypeAttribute):
    name = "pdl_region.region"


PDL_Region = Dialect(
    "pdl_region",
    [
        RegionOp
    ],
    [
        RegionType
    ]
)
