from __future__ import annotations

from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
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


@irdl_op_definition
class NewVarRangeOp(IRDLOperation):
    """Create a new range constraint variable."""

    name = "irdl_interp.new_var.range"
    result = result_def(VariableRangeType)

    assembly_format = "attr-dict"


@irdl_op_definition
class NewVarIntOp(IRDLOperation):
    """Create a new integer constraint variable."""

    name = "irdl_interp.new_var.int"
    result = result_def(VariableIntType)

    assembly_format = "attr-dict"


# ==============================================================================
# Dialect Definition
# ==============================================================================

IRDLInterp = Dialect(
    "irdl_interp",
    [
        NewVarAttrOp,
        NewVarRangeOp,
        NewVarIntOp,
    ],
    [
        VariableAttrType,
        VariableRangeType,
        VariableIntType,
        AttrType,
    ],
)
