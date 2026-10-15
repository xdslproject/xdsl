from __future__ import annotations

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
)

# ==============================================================================
# Types
# ==============================================================================


@irdl_attr_definition
class VariableAttrType(ParametrizedAttribute, TypeAttribute):
    """Constraint variable for attributes."""

    name = "irdl_interp.variable.attr"


@irdl_attr_definition
class VariableRangeType(ParametrizedAttribute, TypeAttribute):
    """Constraint variable for ranges of attributes."""

    name = "irdl_interp.variable.range"


@irdl_attr_definition
class VariableIntType(ParametrizedAttribute, TypeAttribute):
    """Constraint variable for integers."""

    name = "irdl_interp.variable.int"


@irdl_attr_definition
class AttrType(ParametrizedAttribute, TypeAttribute):
    """Wrapper for concrete attribute values."""

    name = "irdl_interp.attr"


# ==============================================================================
# Operations
# ==============================================================================


@irdl_op_definition
class NewVarAttrOp(IRDLOperation):
    """Create a new attribute constraint variable."""

    name = "irdl_interp.new_var.attr"
    result = result_def(VariableAttrType)

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__(result_types=[VariableAttrType()])


@irdl_op_definition
class NewVarRangeOp(IRDLOperation):
    """Create a new range constraint variable."""

    name = "irdl_interp.new_var.range"
    result = result_def(VariableRangeType)

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__(result_types=[VariableRangeType()])


@irdl_op_definition
class NewVarIntOp(IRDLOperation):
    """Create a new integer constraint variable."""

    name = "irdl_interp.new_var.int"
    result = result_def(VariableIntType)

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__(result_types=[VariableIntType()])


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """Wrap a concrete attribute value."""

    name = "irdl_interp.constant.attr"
    value = prop_def(Attribute)
    result = result_def(AttrType)

    assembly_format = "$value attr-dict"

    def __init__(self, value: Attribute):
        super().__init__(
            properties={"value": value},
            result_types=[AttrType()],
        )


# ==============================================================================
# Dialect Definition
# ==============================================================================

IRDLInterp = Dialect(
    "irdl_interp",
    [
        NewVarAttrOp,
        NewVarRangeOp,
        NewVarIntOp,
        ConstantOp,
    ],
    [
        VariableAttrType,
        VariableRangeType,
        VariableIntType,
        AttrType,
    ],
)
