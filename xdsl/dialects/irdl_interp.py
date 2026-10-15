from __future__ import annotations

from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition

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
# Dialect Definition
# ==============================================================================

IRDLInterp = Dialect(
    "irdl_interp",
    [],
    [
        VariableAttrType,
        VariableRangeType,
        VariableIntType,
        AttrType,
    ],
)
