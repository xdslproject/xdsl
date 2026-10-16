from __future__ import annotations

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IndexType,
    IntegerType,
    StringAttr,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
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


@irdl_attr_definition
class OperationType(ParametrizedAttribute, TypeAttribute):
    """Type for operation handles."""

    name = "irdl_interp.operation"


@irdl_attr_definition
class RegionType(ParametrizedAttribute, TypeAttribute):
    """Type for region handles."""

    name = "irdl_interp.region"


@irdl_attr_definition
class BlockType(ParametrizedAttribute, TypeAttribute):
    """Type for block handles (successors)."""

    name = "irdl_interp.block"


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


@irdl_op_definition
class UnifyAttrOp(IRDLOperation):
    """
    Unify two attributes or attribute variables.

    Returns i1: true on success (with binding), false on conflict.
    """

    name = "irdl_interp.unify.attr"
    lhs = operand_def(AttrType)
    rhs = operand_def(AttrType | VariableAttrType)
    result = result_def(IntegerType(1))

    assembly_format = "$lhs `:` type($lhs) `and` $rhs `:` type($rhs) attr-dict"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=[lhs, rhs], result_types=[IntegerType(1)])


@irdl_op_definition
class UnifyIntOp(IRDLOperation):
    """
    Unify two integers or integer variables.

    Returns i1: true on success (with binding), false on conflict.
    """

    name = "irdl_interp.unify.int"
    lhs = operand_def(IndexType)
    rhs = operand_def(IndexType | VariableIntType)
    result = result_def(IntegerType(1))

    assembly_format = "$lhs `:` type($lhs) `and` $rhs `:` type($rhs) attr-dict"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=[lhs, rhs], result_types=[IntegerType(1)])


@irdl_op_definition
class UnifyRangeOp(IRDLOperation):
    """
    Unify two ranges or range variables.

    Returns i1: true on success (with binding), false on conflict.
    """

    name = "irdl_interp.unify.range"
    lhs = operand_def(TensorType[AttrType])
    rhs = operand_def(TensorType[AttrType] | VariableRangeType)
    result = result_def(IntegerType(1))

    assembly_format = "$lhs `:` type($lhs) `and` $rhs `:` type($rhs) attr-dict"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=[lhs, rhs], result_types=[IntegerType(1)])


@irdl_op_definition
class CheckBaseOp(IRDLOperation):
    """Check that an attribute has a specific base type. Returns i1."""

    name = "irdl_interp.check_base"
    attr = operand_def(AttrType | VariableAttrType)
    base = prop_def(StringAttr)
    result = result_def(IntegerType(1))

    assembly_format = "$attr `:` type($attr) `,` $base attr-dict"

    def __init__(self, attr: SSAValue | Operation, base: StringAttr):
        super().__init__(
            operands=[attr], properties={"base": base}, result_types=[IntegerType(1)]
        )


@irdl_op_definition
class AssertOp(IRDLOperation):
    """Assert that a condition is true. Fails with message if false."""

    name = "irdl_interp.assert"
    condition = operand_def(IntegerType(1))
    message = prop_def(StringAttr)

    assembly_format = "$condition `,` $message attr-dict"

    def __init__(self, condition: SSAValue | Operation, message: StringAttr):
        super().__init__(operands=[condition], properties={"message": message})


@irdl_op_definition
class GetPropOp(IRDLOperation):
    """Extract a property from an operation by name."""

    name = "irdl_interp.get_prop"
    op = operand_def(OperationType)
    prop_name = prop_def(StringAttr)
    result = result_def(AttrType)

    assembly_format = "$op `,` $prop_name attr-dict"

    def __init__(self, op: SSAValue | Operation, prop_name: StringAttr):
        super().__init__(
            operands=[op],
            properties={"prop_name": prop_name},
            result_types=[AttrType()],
        )


@irdl_op_definition
class GetOperandTypesOp(IRDLOperation):
    """Extract all operand types as a tensor."""

    name = "irdl_interp.get_operand_types"
    op = operand_def(OperationType)
    result = result_def(TensorType(AttrType(), (DYNAMIC_INDEX,)))

    assembly_format = "$op attr-dict"

    def __init__(self, op: SSAValue | Operation):
        super().__init__(
            operands=[op], result_types=[TensorType(AttrType(), (DYNAMIC_INDEX,))]
        )


@irdl_op_definition
class GetResultTypesOp(IRDLOperation):
    """Extract all result types as a tensor."""

    name = "irdl_interp.get_result_types"
    op = operand_def(OperationType)
    result = result_def(TensorType(AttrType(), (DYNAMIC_INDEX,)))

    assembly_format = "$op attr-dict"

    def __init__(self, op: SSAValue | Operation):
        super().__init__(
            operands=[op], result_types=[TensorType(AttrType(), (DYNAMIC_INDEX,))]
        )


@irdl_op_definition
class GetParamsOp(IRDLOperation):
    """Extract all parameters from a parametrized attribute."""

    name = "irdl_interp.get_params"
    attr = operand_def(AttrType)
    result = result_def(TensorType(AttrType(), (DYNAMIC_INDEX,)))

    assembly_format = "$attr attr-dict"

    def __init__(self, attr: SSAValue | Operation):
        super().__init__(
            operands=[attr], result_types=[TensorType(AttrType(), (DYNAMIC_INDEX,))]
        )


@irdl_op_definition
class GetPropCountOp(IRDLOperation):
    """Get the number of properties on an operation."""

    name = "irdl_interp.get_prop_count"
    op = operand_def(OperationType)
    result = result_def(IndexType)

    assembly_format = "$op attr-dict"

    def __init__(self, op: SSAValue | Operation):
        super().__init__(
            operands=[op],
            result_types=[IndexType()],
        )


@irdl_op_definition
class GetRegionsOp(IRDLOperation):
    """Extract all regions as a tensor."""

    name = "irdl_interp.get_regions"
    op = operand_def(OperationType)
    result = result_def(TensorType(RegionType(), (DYNAMIC_INDEX,)))

    assembly_format = "$op attr-dict"

    def __init__(self, op: SSAValue | Operation):
        super().__init__(
            operands=[op], result_types=[TensorType(RegionType(), (DYNAMIC_INDEX,))]
        )


@irdl_op_definition
class GetSuccessorsOp(IRDLOperation):
    """Extract all successors as a tensor."""

    name = "irdl_interp.get_successors"
    op = operand_def(OperationType)
    result = result_def(TensorType(BlockType(), (DYNAMIC_INDEX,)))

    assembly_format = "$op attr-dict"

    def __init__(self, op: SSAValue | Operation):
        super().__init__(
            operands=[op], result_types=[TensorType(BlockType(), (DYNAMIC_INDEX,))]
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
        UnifyAttrOp,
        UnifyIntOp,
        UnifyRangeOp,
        CheckBaseOp,
        AssertOp,
        GetPropOp,
        GetOperandTypesOp,
        GetResultTypesOp,
        GetParamsOp,
        GetPropCountOp,
        GetRegionsOp,
        GetSuccessorsOp,
    ],
    [
        VariableAttrType,
        VariableRangeType,
        VariableIntType,
        AttrType,
        OperationType,
        RegionType,
        BlockType,
    ],
)
