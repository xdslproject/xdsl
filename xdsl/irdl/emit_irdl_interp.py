"""
Emit irdl_interp IR for constraint verification.

This module contains the logic for lowering constraint classes to irdl_interp operations.
It is separate from constraints.py to avoid circular imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.arith import OrIOp
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.irdl_interp import AssertOp, CheckBaseOp, ConstantOp, UnifyAttrOp
from xdsl.dialects.scf import IfOp, YieldOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.irdl import (
    AnyOf,
    AttrSetConstraint,
    BaseAttr,
    EqAttrConstraint,
    VarConstraint,
)


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


def emit_any_of_constraint_irdl_interp(
    constraint: AnyOf,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
) -> None:
    """
    Emit irdl_interp IR to verify an AnyOf constraint using scf.if dispatch.

    Generates nested scf.if operations that check the attribute's base type
    and dispatch to the appropriate constraint:

      %is_base0 = irdl_interp.check_base %value, "BaseType0"
      scf.if %is_base0 {
        // Check constraint for base0
        irdl_interp.check_base %value, "ExpectedType"
        irdl_interp.assert ...
        scf.yield
      } else {
        %is_base1 = irdl_interp.check_base %value, "BaseType1"
        scf.if %is_base1 {
          // Check constraint for base1
          scf.yield
        } else {
          // Abstract constraint or fail
          irdl_interp.assert false, "Expected one of ..."
          scf.yield
        }
      }

    Args:
        constraint: The AnyOf constraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
        context: Immutable context containing all constraint variables
    """
    from xdsl.irdl import BaseAttr

    # Build error message with all expected types
    expected_types: list[str] = []
    for base_type, constr in constraint._based_constrs.items():  # pyright: ignore[reportPrivateUsage]
        if isinstance(constr, BaseAttr):
            expected_types.append(constr.attr.__name__)
    abstr_constr = constraint._abstr_constr  # pyright: ignore[reportPrivateUsage]
    if abstr_constr is not None and isinstance(abstr_constr, BaseAttr):
        expected_types.append(abstr_constr.attr.__name__)
    error_msg = StringAttr(f"Expected one of {', '.join(expected_types)}")

    # Build nested scf.if chain from innermost to outermost
    # Start with the innermost else block (abstract constraint or fail)
    innermost_else_block = Block()
    innermost_builder = Builder(InsertPoint.at_end(innermost_else_block))
    if abstr_constr is not None:
        # Emit abstract constraint check
        abstr_constr.emit_irdl_interp(value, innermost_builder, context)
    else:
        # No abstract constraint - always fail
        false_const = ArithConstantOp(IntegerAttr.from_bool(False))
        innermost_builder.insert(false_const)
        assert_op = AssertOp(false_const.result, error_msg)
        innermost_builder.insert(assert_op)
    yield_op = YieldOp()
    innermost_builder.insert(yield_op)

    innermost_else_region = Region([innermost_else_block])

    # Build the chain of scf.if operations from last to first
    current_else_region = innermost_else_region
    based_constrs_list = list(constraint._based_constrs.items())  # pyright: ignore[reportPrivateUsage]

    for base_type, attr_constr in reversed(based_constrs_list):
        # Create the true region that checks this specific constraint
        true_block = Block()
        true_builder = Builder(InsertPoint.at_end(true_block))
        attr_constr.emit_irdl_interp(value, true_builder, context)
        yield_op = YieldOp()
        true_builder.insert(yield_op)

        true_region = Region([true_block])

        # Get the base type name for check_base
        base_type_name = StringAttr(base_type.__name__)

        # Create a block to hold the check and if
        outer_block = Block()
        outer_builder = Builder(InsertPoint.at_end(outer_block))

        # Insert the check
        check_op = CheckBaseOp(value, base_type_name)
        outer_builder.insert(check_op)

        # Insert the if
        if_op = IfOp(
            check_op.result,
            [],
            true_region,
            current_else_region,
        )
        outer_builder.insert(if_op)

        yield_op = YieldOp()
        outer_builder.insert(yield_op)

        current_else_region = Region([outer_block])

    # Now insert the outermost region's operations into the actual builder
    # The outermost region has a single block with check + if + yield
    outermost_block = current_else_region.blocks[0]
    # Move operations from the temporary block to the actual builder's block
    for op in list(outermost_block.ops):
        op.detach()
        builder.insert(op)
