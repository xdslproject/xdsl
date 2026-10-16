"""
Tests for emit_irdl_interp functionality.
"""

from xdsl.builder import Builder
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.irdl_interp import (
    AssertOp,
    CheckBaseOp,
    ConstantOp,
    NewVarAttrOp,
    UnifyAttrOp,
)
from xdsl.ir import Block
from xdsl.irdl import AnyAttr, BaseAttr, EqAttrConstraint, VarConstraint
from xdsl.irdl.emit_irdl_interp import EmitContext
from xdsl.rewriter import InsertPoint


def test_emit_base_attr_irdl_interp():
    """Test that BaseAttr constraint lowering generates correct IR."""
    # Create a module to build in
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)

    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with
    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Create empty context
    context = EmitContext()

    # Emit irdl_interp IR
    constraint = BaseAttr(IntegerAttr)
    constraint.emit_irdl_interp(constant_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: ConstantOp, CheckBaseOp, AssertOp
    assert len(ops) == 3

    # Check that CheckBaseOp was created
    check_op = ops[1]
    assert isinstance(check_op, CheckBaseOp)
    assert check_op.attr is constant_op.result
    assert check_op.base.data == "integer"

    # Check that AssertOp was created
    assert_op = ops[2]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is check_op.result
    assert "Expected integer" in assert_op.message.data


def test_any_attr_emit_is_noop():
    """Test that AnyAttr.emit_irdl_interp is a no-op."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Create empty context
    context = EmitContext()

    # Count ops before
    ops_before = len(list(block.ops))

    # Call emit_irdl_interp - should be a no-op
    any_attr = AnyAttr()
    any_attr.emit_irdl_interp(constant_op.result, builder, context)

    # Count ops after - should be the same
    ops_after = len(list(block.ops))
    assert ops_before == ops_after, (
        "AnyAttr.emit_irdl_interp should not generate any IR"
    )


def test_var_constraint_emit_with_any_attr():
    """Test that VarConstraint.emit_irdl_interp with AnyAttr inner constraint generates only unify."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    # Create a variable first (simulating Phase 2 setup)
    var_op = NewVarAttrOp()
    builder.insert(var_op)

    # Create a constant attribute value to test with
    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Create context with the variable
    context = EmitContext(_attr_variables={"T": var_op.result})

    # Create VarConstraint with AnyAttr inner constraint (no checks, just unify)
    var_constraint = VarConstraint("T", AnyAttr())
    var_constraint.emit_irdl_interp(constant_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: NewVarAttrOp, ConstantOp, UnifyAttrOp (no CheckBaseOp/AssertOp from AnyAttr)
    assert len(ops) == 3

    # Check that UnifyAttrOp was created
    unify_op = ops[2]
    assert isinstance(unify_op, UnifyAttrOp)
    assert unify_op.lhs is constant_op.result
    assert unify_op.rhs is var_op.result


def test_var_constraint_emit_with_base_attr():
    """Test that VarConstraint.emit_irdl_interp with BaseAttr inner constraint generates check + unify."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    # Create a variable first (simulating Phase 2 setup)
    var_op = NewVarAttrOp()
    builder.insert(var_op)

    # Create a constant attribute value to test with
    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Create context with the variable
    context = EmitContext(_attr_variables={"T": var_op.result})

    # Create VarConstraint with BaseAttr inner constraint
    var_constraint = VarConstraint("T", BaseAttr(IntegerAttr))
    var_constraint.emit_irdl_interp(constant_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: NewVarAttrOp, ConstantOp, CheckBaseOp, AssertOp, UnifyAttrOp
    assert len(ops) == 5

    # Check that CheckBaseOp was created (from BaseAttr inner constraint)
    check_op = ops[2]
    assert isinstance(check_op, CheckBaseOp)
    assert check_op.attr is constant_op.result
    assert check_op.base.data == "integer"

    # Check that AssertOp was created (from BaseAttr inner constraint)
    assert_op = ops[3]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is check_op.result
    assert "Expected integer" in assert_op.message.data

    # Check that UnifyAttrOp was created
    unify_op = ops[4]
    assert isinstance(unify_op, UnifyAttrOp)
    assert unify_op.lhs is constant_op.result
    assert unify_op.rhs is var_op.result


def test_eq_attr_constraint_emit():
    """Test that EqAttrConstraint.emit_irdl_interp generates constant + unify."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context (EqAttrConstraint doesn't need variables)
    context = EmitContext()

    # Create EqAttrConstraint with a specific attribute value
    expected_attr = IntegerAttr.from_index_int_value(42)
    eq_constraint = EqAttrConstraint(expected_attr)
    eq_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: value ConstantOp, constraint ConstantOp, UnifyAttrOp
    assert len(ops) == 3

    # Check that ConstantOp was created for the constraint
    constraint_constant_op = ops[1]
    assert isinstance(constraint_constant_op, ConstantOp)
    assert constraint_constant_op.value == expected_attr

    # Check that UnifyAttrOp was created
    unify_op = ops[2]
    assert isinstance(unify_op, UnifyAttrOp)
    assert unify_op.lhs is value_op.result
    assert unify_op.rhs is constraint_constant_op.result
