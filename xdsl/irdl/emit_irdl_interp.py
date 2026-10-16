"""
Emit irdl_interp IR for constraint verification.

This module contains the logic for lowering constraint classes to irdl_interp operations.
It is separate from constraints.py to avoid circular imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.irdl_interp import AssertOp, CheckBaseOp
from xdsl.ir import SSAValue
from xdsl.irdl import BaseAttr


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
