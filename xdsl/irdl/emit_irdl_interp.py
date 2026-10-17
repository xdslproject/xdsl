"""
Emit irdl_interp IR for constraint verification.

This module contains the logic for lowering constraint classes to irdl_interp operations.
It is separate from constraints.py to avoid circular imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from typing_extensions import TypeVar

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.arith import OrIOp
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.irdl_interp import (
    AssertOp,
    AttrType,
    CheckBaseOp,
    ConstantOp,
    GetParamsOp,
    UnifyAttrOp,
    UnifyIntOp,
)
from xdsl.dialects.scf import IfOp, YieldOp
from xdsl.dialects.tensor import DimOp, ExtractOp
from xdsl.ir import Block, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import (
    AllOf,
    AnyOf,
    AttrSetConstraint,
    BaseAttr,
    EqAttrConstraint,
    ParamAttrConstraint,
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
    message: str = "",
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
        message: Optional message prefix to prepend to the default message
    """
    # Get the type name for the constraint
    type_name = constraint.attr.name

    # Create the check_base operation
    check_op = CheckBaseOp(value, StringAttr(type_name))
    builder.insert(check_op)

    # Create the assert operation with default message, prepended by message prefix if provided
    full_message = f"{message}Expected {type_name}"
    assert_op = AssertOp(check_op.result, full_message)
    builder.insert(assert_op)


def emit_var_constraint_irdl_interp(
    constraint: VarConstraint,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
    message: str = "",
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
        message: Optional message prefix to prepend to default messages
    """
    # Get the variable SSA value from context
    var_ssa = context.attr_variables[constraint.name]

    # Check the inner constraint on the value with the message prefix
    constraint.constraint.emit_irdl_interp(value, builder, context, message)

    # Unify the value with the variable and assert
    unify_op = UnifyAttrOp(value, var_ssa)
    builder.insert(unify_op)
    full_message = f"{message}Expected to unify with variable {constraint.name}"
    assert_op = AssertOp(unify_op.result, full_message)
    builder.insert(assert_op)


def emit_eq_attr_constraint_irdl_interp(
    constraint: EqAttrConstraint,
    value: SSAValue,
    builder: Builder,
    message: str = "",
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
        message: Optional message prefix to prepend to the default message
    """
    # Create a constant operation with the attribute value
    constant_op = ConstantOp(constraint.attr)
    builder.insert(constant_op)

    # Unify the value with the constant
    unify_op = UnifyAttrOp(value, constant_op.result)
    builder.insert(unify_op)
    full_message = f"{message}Expected {constraint.attr}"
    assert_op = AssertOp(unify_op.result, full_message)
    builder.insert(assert_op)


def emit_attr_set_constraint_irdl_interp(
    constraint: AttrSetConstraint,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
    message: str = "",
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
        message: Optional message prefix to prepend to the default message
    """
    # Start with false constant - will be ORed with all unify results
    false_const = ArithConstantOp(IntegerAttr.from_bool(False))
    builder.insert(false_const)
    accum_result = false_const.result

    # Sort values for deterministic order
    sorted_values = sorted(((v, str(v)) for v in constraint.values), key=lambda x: x[1])

    # Create constants for each attribute in the set and unify
    for attr, _ in sorted_values:
        const_op = ConstantOp(attr)
        builder.insert(const_op)

        unify_op = UnifyAttrOp(value, const_op.result)
        builder.insert(unify_op)

        # OR the unify result with the accumulator
        or_op = OrIOp(accum_result, unify_op.result)
        builder.insert(or_op)
        accum_result = or_op.result

    # Assert the final result with default message, prepended by message prefix if provided
    if len(constraint.values) == 0:
        full_message = f"{message}Expected one of (empty set)"
    else:
        attrs_str = ", ".join(s for _, s in sorted_values)
        full_message = f"{message}Expected one of {attrs_str}"
    assert_op = AssertOp(accum_result, full_message)
    builder.insert(assert_op)


def emit_any_of_constraint_irdl_interp(
    constraint: AnyOf,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
    message: str = "",
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
    based_constrs = constraint._based_constrs  # pyright: ignore[reportPrivateUsage]
    abstr_constr = constraint._abstr_constr  # pyright: ignore[reportPrivateUsage]

    current_builder = builder
    for base_type, attr_constr in based_constrs.items():
        # Create the true region that checks this specific constraint
        true_block = Block()
        true_builder = Builder(InsertPoint.at_end(true_block))
        true_builder.extend_from_listener(current_builder)
        attr_constr.emit_irdl_interp(value, true_builder, context, message)
        true_builder.insert(YieldOp())

        true_region = Region([true_block])

        # Get the base type name for check_base
        base_type_name = StringAttr(base_type.name)

        false_block = Block()
        false_region = Region([false_block])

        # Insert the check
        check_op = current_builder.insert(CheckBaseOp(value, base_type_name))

        # Insert the if
        current_builder.insert(
            IfOp(
                check_op.result,
                [],
                true_region,
                false_region,
            )
        )

        false_builder = Builder(InsertPoint.at_start(false_block))
        false_builder.extend_from_listener(current_builder)
        false_builder.insert(YieldOp())
        false_builder.insertion_point = InsertPoint.at_start(false_block)
        current_builder = false_builder

    # Deal with the abstract constraint
    if abstr_constr is not None:
        abstr_constr.emit_irdl_interp(value, current_builder, context, message)
    else:
        # No abstract constraint - always fail
        false_const = ArithConstantOp(IntegerAttr.from_bool(False))
        current_builder.insert(false_const)
        full_message = f"{message}Unexpected attribute"
        current_builder.insert(AssertOp(false_const.result, full_message))


_ParametrizedAttributeInvT = TypeVar(
    "_ParametrizedAttributeInvT", bound=ParametrizedAttribute
)


def emit_allof_constraint_irdl_interp(
    constraint: AllOf,
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
    message: str = "",
) -> None:
    """
    Emit irdl_interp IR to verify an AllOf constraint.

    Generates sequential checks for all constraints:

      // Check first constraint
      irdl_interp.check_base %value, "Type1"
      irdl_interp.assert ...
      // Check second constraint
      irdl_interp.check_base %value, "Type2"
      irdl_interp.assert ...
      ...

    All constraints must pass for the AllOf to succeed.

    Args:
        constraint: The AllOf constraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
        context: Immutable context containing all constraint variables
        message: Optional message prefix to prepend to default messages
    """
    # Emit all constraints sequentially
    for attr_constr in constraint.attr_constrs:
        attr_constr.emit_irdl_interp(value, builder, context, message)


def emit_param_attr_constraint_irdl_interp(
    constraint: ParamAttrConstraint[_ParametrizedAttributeInvT],
    value: SSAValue,
    builder: Builder,
    context: EmitContext,
    message: str = "",
) -> None:
    """
    Emit irdl_interp IR to verify a ParamAttrConstraint.

    Generates:
      %check = irdl_interp.check_base %value, "TypeName" : i1
      irdl_interp.assert %check, "Expected TypeName"
      %params = irdl_interp.get_params %value
      %param_count = tensor.dim %params, %c0
      irdl_interp.unify.int %param_count : index and %expected_count : index
      %p0 = tensor.extract %params[0] : tensor<?x!irdl_interp.attr>
      // Check p0 with its constraint
      %p1 = tensor.extract %params[1] : tensor<?x!irdl_interp.attr>
      // Check p1 with its constraint
      ...

    Args:
        constraint: The ParamAttrConstraint to lower
        value: The SSA value to verify (should be !irdl_interp.attr type)
        builder: The builder to use for inserting operations
        context: Immutable context containing all constraint variables
        message: Optional message prefix to prepend to default messages
    """
    # message already defaults to ""

    # Get the base type name for the constraint
    type_name = constraint.base_attr.name

    # Check base type
    check_op = CheckBaseOp(value, StringAttr(type_name))
    builder.insert(check_op)
    type_message = f"{message}Expected {type_name}"
    assert_op = AssertOp(check_op.result, type_message)
    builder.insert(assert_op)

    # Get all parameters
    params_op = GetParamsOp(value)
    builder.insert(params_op)

    # Check parameter count
    c0 = ArithConstantOp(IntegerAttr.from_index_int_value(0))
    builder.insert(c0)
    param_count_op = DimOp(params_op.result, c0.result)
    builder.insert(param_count_op)
    expected_count = ArithConstantOp(
        IntegerAttr.from_index_int_value(len(constraint.param_constrs))
    )
    builder.insert(expected_count)
    unify_op = UnifyIntOp(param_count_op.result, expected_count.result)
    builder.insert(unify_op)
    count_message = f"{message}Expected {len(constraint.param_constrs)} parameters"
    assert_op = AssertOp(unify_op.result, count_message)
    builder.insert(assert_op)

    # Extract and check each parameter
    for idx, param_constr in enumerate(constraint.param_constrs):
        # Extract parameter at index idx
        idx_op = ArithConstantOp(IntegerAttr.from_index_int_value(idx))
        builder.insert(idx_op)
        extract_op = ExtractOp(params_op.result, [idx_op.result], AttrType())
        builder.insert(extract_op)

        # Check parameter with its constraint
        param_constr.emit_irdl_interp(extract_op.result, builder, context, message)
