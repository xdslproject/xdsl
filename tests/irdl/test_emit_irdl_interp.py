"""
Tests for emit_irdl_interp functionality.
"""

from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.arith import OrIOp
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp
from xdsl.dialects.irdl_interp import (
    AssertOp,
    CheckBaseOp,
    ConstantOp,
    NewVarAttrOp,
    UnifyAttrOp,
)
from xdsl.ir import Block
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    AttrSetConstraint,
    BaseAttr,
    EqAttrConstraint,
    VarConstraint,
)
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
    """Test that VarConstraint.emit_irdl_interp with AnyAttr inner constraint generates unify + assert."""
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

    # Should have: NewVarAttrOp, ConstantOp, UnifyAttrOp, AssertOp
    assert len(ops) == 4

    # Check that UnifyAttrOp was created
    unify_op = ops[2]
    assert isinstance(unify_op, UnifyAttrOp)
    assert unify_op.lhs is constant_op.result
    assert unify_op.rhs is var_op.result

    # Check that AssertOp was created
    assert_op = ops[3]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is unify_op.result
    assert "Expected to unify with variable T" in assert_op.message.data


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

    # Should have: NewVarAttrOp, ConstantOp, CheckBaseOp, AssertOp (base), UnifyAttrOp, AssertOp (unify)
    assert len(ops) == 6

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

    # Check that AssertOp was created (from unify)
    unify_assert_op = ops[5]
    assert isinstance(unify_assert_op, AssertOp)
    assert unify_assert_op.condition is unify_op.result
    assert "Expected to unify with variable T" in unify_assert_op.message.data


def test_eq_attr_constraint_emit():
    """Test that EqAttrConstraint.emit_irdl_interp generates constant + unify + assert."""
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

    # Should have: value ConstantOp, constraint ConstantOp, UnifyAttrOp, AssertOp
    assert len(ops) == 4

    # Check that ConstantOp was created for the constraint
    constraint_constant_op = ops[1]
    assert isinstance(constraint_constant_op, ConstantOp)
    assert constraint_constant_op.value == expected_attr

    # Check that UnifyAttrOp was created
    unify_op = ops[2]
    assert isinstance(unify_op, UnifyAttrOp)
    assert unify_op.lhs is value_op.result
    assert unify_op.rhs is constraint_constant_op.result

    # Check that AssertOp was created
    assert_op = ops[3]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is unify_op.result
    assert f"Expected {expected_attr}" in assert_op.message.data


def test_attr_set_constraint_emit_single():
    """Test that AttrSetConstraint with single value generates false + constant + unify + OR + assert."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create AttrSetConstraint with a single attribute value
    expected_attr = IntegerAttr.from_index_int_value(42)
    set_constraint = AttrSetConstraint(frozenset([expected_attr]))
    set_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: value ConstantOp, arith.constant false, constraint ConstantOp, UnifyAttrOp, OrIOp, AssertOp
    assert len(ops) == 6

    # Check that arith.constant false was created
    false_const = ops[1]
    assert isinstance(false_const, ArithConstantOp)

    # Check that ConstantOp was created for the constraint
    constraint_constant_op = ops[2]
    assert isinstance(constraint_constant_op, ConstantOp)
    assert constraint_constant_op.value == expected_attr

    # Check that UnifyAttrOp was created
    unify_op = ops[3]
    assert isinstance(unify_op, UnifyAttrOp)
    assert unify_op.lhs is value_op.result
    assert unify_op.rhs is constraint_constant_op.result

    # Check that OrIOp was created
    or_op = ops[4]
    assert isinstance(or_op, OrIOp)

    # Check that AssertOp was created
    assert_op = ops[5]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is or_op.result
    assert "Expected one of" in assert_op.message.data


def test_attr_set_constraint_emit_multiple():
    """Test that AttrSetConstraint with multiple values generates false + constants + unify + OR + assert."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create AttrSetConstraint with multiple attribute values
    attr1 = IntegerAttr.from_index_int_value(42)
    attr2 = IntegerAttr.from_index_int_value(43)
    set_constraint = AttrSetConstraint(frozenset([attr1, attr2]))
    set_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: value ConstantOp, arith.constant false,
    # then for each attr: ConstantOp, UnifyAttrOp, OrIOp, then AssertOp
    # Total: 1 (value) + 1 (false) + 3*2 (const+unify+or for each attr) + 1 (assert) = 9
    assert len(ops) == 9

    # Check that arith.constant false was created
    false_const = ops[1]
    assert isinstance(false_const, ArithConstantOp)

    # Check the pattern: for each attr we have ConstantOp, UnifyAttrOp, OrIOp
    # ops[2], ops[3], ops[4] = first attr's const, unify, or
    # ops[5], ops[6], ops[7] = second attr's const, unify, or
    assert isinstance(ops[2], ConstantOp)
    assert isinstance(ops[3], UnifyAttrOp)
    assert isinstance(ops[4], OrIOp)
    assert isinstance(ops[5], ConstantOp)
    assert isinstance(ops[6], UnifyAttrOp)
    assert isinstance(ops[7], OrIOp)

    # Check that AssertOp was created (last op)
    assert_op = ops[8]
    assert isinstance(assert_op, AssertOp)
    assert "Expected one of" in assert_op.message.data


def test_attr_set_constraint_emit_empty():
    """Test that AttrSetConstraint with empty set generates false constant + assert."""
    module = ModuleOp([])
    block = Block()
    module.body.add_block(block)
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create AttrSetConstraint with empty set
    set_constraint: AttrConstraint = AttrSetConstraint(frozenset({}))
    set_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Verify the operations were created
    ops = list(block.ops)

    # Should have: value ConstantOp, arith.constant false, AssertOp
    assert len(ops) == 3

    # Check that arith.constant false was created (i1 type)
    false_const = ops[1]
    assert isinstance(false_const, ArithConstantOp)
    assert false_const.result.type == IntegerType(1)

    # Check that AssertOp was created
    assert_op = ops[2]
    assert isinstance(assert_op, AssertOp)
    assert assert_op.condition is false_const.result
    assert "Expected one of (empty set)" in assert_op.message.data
