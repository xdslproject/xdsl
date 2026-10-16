"""
Emit irdl_interp IR for constraint verification.

This module contains the logic for lowering constraint classes to irdl_interp operations.
It is separate from constraints.py to avoid circular imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.arith import OrIOp
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.irdl_interp import AssertOp, CheckBaseOp, ConstantOp, UnifyAttrOp
from xdsl.ir import SSAValue
from xdsl.irdl import AttrSetConstraint, BaseAttr, EqAttrConstraint, VarConstraint


@dataclass(frozen=True)
class EmitContext:
    """
    Immutable context for emit_irdl_interp operations.

    Created during variable setup phase, then used read-only during
    constraint lowering.
    """

    _attr_variables: dict[str, SSAValue] = field(default_factory=dict, repr=False)
    _range_variables: dict[str, SSAValue] = field(default_factory=dict, repr=False)
    _int_variables: dict[str, SSAValue] = field(default_factory=dict, repr=False)

    @property
    def attr_variables(self) -> Mapping[str, SSAValue]:
        """Read-only view of attribute variables."""
        return self._attr_variables

    @property
    def range_variables(self) -> Mapping[str, SSAValue]:
        """Read-only view of range variables."""
        return self._range_variables

    @property
    def int_variables(self) -> Mapping[str, SSAValue]:
        """Read-only view of integer variables."""
        return self._int_variables


def emit_base_attr_irdl_interp(
    constraint: BaseAttr,
    value: SSAValue,
    builder: Builder,
) -> None:
    """
    Emit irdl_interp IR to verify a BaseAttr constraint.

    Generates:
      %check = irdl_interp.check_base %value, "TypeName" : i1
      irdl_interp.assert %check, "Expected TypeName"

    Args:
        constraint: The BaseAttr constraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
    """
    # Get the type name for the constraint
    type_name = (
        constraint.attr.name
        if hasattr(constraint.attr, "name")
        else constraint.attr.__name__
    )

    # Create the check_base operation
    check_op = CheckBaseOp(value, StringAttr(type_name))
    builder.insert(check_op)

    # Create the assert operation
    message = StringAttr(f"Expected {type_name}")
    assert_op = AssertOp(check_op.result, message)
    builder.insert(assert_op)


def emit_var_constraint_irdl_interp(
    constraint: VarConstraint,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
) -> None:
    """
    Emit irdl_interp IR to verify a VarConstraint.

    Generates:
      // First, check the inner constraint on the value
      %check = irdl_interp.check_base %value, "TypeName" : i1
      irdl_interp.assert %check, "Expected TypeName"
      // Then, unify the value with the variable
      irdl_interp.unify.attr %value : !irdl_interp.attr and %T : !irdl_interp.variable.attr

    Args:
        constraint: The VarConstraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
        context: Immutable context containing all constraint variables
    """
    # Get the variable SSA value from context
    var_ssa = context.attr_variables[constraint.name]

    # Check the inner constraint on the value
    constraint.constraint.emit_irdl_interp(value, builder, context)

    # Unify the value with the variable and assert
    unify_op = UnifyAttrOp(value, var_ssa)
    builder.insert(unify_op)
    assert_op = AssertOp(
        unify_op.result,
        StringAttr(f"Expected to unify with variable {constraint.name}"),
    )
    builder.insert(assert_op)


def emit_eq_attr_constraint_irdl_interp(
    constraint: EqAttrConstraint,
    value: SSAValue,
    builder: Builder,
) -> None:
    """
    Emit irdl_interp IR to verify an EqAttrConstraint.

    Generates:
      %concrete = irdl_interp.constant.attr <attr>
      irdl_interp.unify.attr %value : !irdl_interp.attr and %concrete : !irdl_interp.attr

    Args:
        constraint: The EqAttrConstraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
    """
    # Create a constant operation with the attribute value
    constant_op = ConstantOp(constraint.attr)
    builder.insert(constant_op)

    # Unify the value with the constant
    unify_op = UnifyAttrOp(value, constant_op.result)
    builder.insert(unify_op)
    assert_op = AssertOp(unify_op.result, StringAttr(f"Expected {constraint.attr}"))
    builder.insert(assert_op)


def emit_attr_set_constraint_irdl_interp(
    constraint: AttrSetConstraint,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
) -> None:
    """
    Emit irdl_interp IR to verify an AttrSetConstraint.

    Generates:
      %false = arith.constant false
      %c0 = irdl_interp.constant.attr <attr0>
      %eq0 = irdl_interp.unify.attr %value, %c0
      %or0 = arith.ori %false, %eq0
      %c1 = irdl_interp.constant.attr <attr1>
      %eq1 = irdl_interp.unify.attr %value, %c1
      %or1 = arith.ori %or0, %eq1
      irdl_interp.assert %or1, "Expected one of {attrs}"

    Note: Always starts with false and ORs all unify results. The `or false x`
    pattern will be canonicalized away by the compiler.

    Args:
        constraint: The AttrSetConstraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
        context: Immutable context containing all constraint variables
    """
    # Start with false constant - will be ORed with all unify results
    false_const = ArithConstantOp(IntegerAttr.from_bool(False))
    builder.insert(false_const)
    accum_result = false_const.result

    # Create constants for each attribute in the set and unify
    for attr in constraint.values:
        const_op = ConstantOp(attr)
        builder.insert(const_op)

        unify_op = UnifyAttrOp(value, const_op.result)
        builder.insert(unify_op)

        # OR the unify result with the accumulator
        or_op = OrIOp(accum_result, unify_op.result)
        builder.insert(or_op)
        accum_result = or_op.result

    # Assert the final result
    if len(constraint.values) == 0:
        message = StringAttr("Expected one of (empty set)")
    else:
        attrs_str = ", ".join(sorted(str(attr) for attr in constraint.values))
        message = StringAttr(f"Expected one of {attrs_str}")
    assert_op = AssertOp(accum_result, message)
    builder.insert(assert_op)
