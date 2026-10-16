"""
Tests for emit_irdl_interp functionality.

Uses is_structurally_equivalent to compare generated IR against expected IR.
"""

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr
from xdsl.dialects.irdl_interp import ConstantOp, IRDLInterp, NewVarAttrOp
from xdsl.ir import Attribute
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    AttrConstraint,
    AttrSetConstraint,
    BaseAttr,
    EqAttrConstraint,
    VarConstraint,
)
from xdsl.irdl.emit_irdl_interp import EmitContext
from xdsl.parser import Parser
from xdsl.rewriter import InsertPoint


def _parse_module(ir: str) -> ModuleOp:
    """Parse IR string into a ModuleOp."""
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(IRDLInterp)
    ctx.load_dialect(scf.Scf)
    parser = Parser(ctx, ir)
    return parser.parse_module()


def test_emit_base_attr_irdl_interp():
    """Test that BaseAttr constraint lowering generates correct IR."""
    # Create a module to build in
    module = ModuleOp([])
    block = module.body.block

    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with
    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Create empty context
    context = EmitContext()

    # Emit irdl_interp IR
    constraint = BaseAttr(IntegerAttr)
    constraint.emit_irdl_interp(constant_op.result, builder, context)

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      irdl_interp.assert %1, "Expected integer"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_any_attr_emit_is_noop():
    """Test that AnyAttr.emit_irdl_interp is a no-op."""
    module = ModuleOp([])
    block = module.body.block
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
    assert ops_before == ops_after


def test_var_constraint_emit_with_any_attr():
    """Test that VarConstraint.emit_irdl_interp with AnyAttr inner constraint generates unify + assert."""
    module = ModuleOp([])
    block = module.body.block
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

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.new_var.attr
      %1 = irdl_interp.constant.attr 42 : index
      %2 = irdl_interp.unify.attr %1 : !irdl_interp.attr and %0 : !irdl_interp.variable.attr
      irdl_interp.assert %2, "Expected to unify with variable T"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_var_constraint_emit_with_base_attr():
    """Test that VarConstraint.emit_irdl_interp with BaseAttr inner constraint generates check + unify."""
    module = ModuleOp([])
    block = module.body.block
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

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.new_var.attr
      %1 = irdl_interp.constant.attr 42 : index
      %2 = irdl_interp.check_base %1 : !irdl_interp.attr, "integer"
      irdl_interp.assert %2, "Expected integer"
      %3 = irdl_interp.unify.attr %1 : !irdl_interp.attr and %0 : !irdl_interp.variable.attr
      irdl_interp.assert %3, "Expected to unify with variable T"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_eq_attr_constraint_emit():
    """Test that EqAttrConstraint.emit_irdl_interp generates constant + unify + assert."""
    module = ModuleOp([])
    block = module.body.block
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

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.constant.attr 42 : index
      %2 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %1 : !irdl_interp.attr
      irdl_interp.assert %2, "Expected 42 : index"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_attr_set_constraint_emit_single():
    """Test that AttrSetConstraint with single value generates false + constant + unify + OR + assert."""
    module = ModuleOp([])
    block = module.body.block
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

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      %2 = irdl_interp.constant.attr 42 : index
      %3 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %2 : !irdl_interp.attr
      %4 = arith.ori %1, %3 : i1
      irdl_interp.assert %4, "Expected one of 42 : index"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_attr_set_constraint_emit_multiple():
    """Test that AttrSetConstraint with multiple values generates false + constants + unify + OR + assert."""
    module = ModuleOp([])
    block = module.body.block
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

    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      %2 = irdl_interp.constant.attr 42 : index
      %3 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %2 : !irdl_interp.attr
      %4 = arith.ori %1, %3 : i1
      %5 = irdl_interp.constant.attr 43 : index
      %6 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %5 : !irdl_interp.attr
      %7 = arith.ori %4, %6 : i1
      irdl_interp.assert %7, "Expected one of 42 : index, 43 : index"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_attr_set_constraint_emit_empty():
    """Test that AttrSetConstraint with empty set generates false constant + assert."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create AttrSetConstraint with empty set
    set_constraint: AttrConstraint = AttrSetConstraint(frozenset({}))
    set_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      irdl_interp.assert %1, "Expected one of (empty set)"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_any_of_constraint_emit():
    """Test that AnyOf constraint lowering generates scf.if dispatch."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create AnyOf constraint with two BaseAttr constraints
    any_of_constraint = AnyOf.get(BaseAttr(IntegerAttr), BaseAttr(StringAttr))
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      scf.if %1 {
        %2 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
        irdl_interp.assert %2, "Expected integer"
      } else {
        %3 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
        scf.if %3 {
          %4 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
          irdl_interp.assert %4, "Expected string"
        } else {
          %5 = arith.constant false
          irdl_interp.assert %5, "Unexpected attribute"
        }
      }
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_any_of_constraint_emit_empty():
    """Test that empty AnyOf constraint lowering generates immediate failure."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create empty AnyOf constraint
    any_of_constraint = AnyOf.get()
    assert isinstance(any_of_constraint, AnyOf)
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      irdl_interp.assert %1, "Unexpected attribute"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


class AbstractTestAttr(Attribute):
    name = "test.abstract"


def test_any_of_constraint_emit_with_abstract():
    """Test that AnyOf constraint with abstract constraint generates proper dispatch."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create empty context
    context = EmitContext()

    # Create AnyOf with one concrete and one abstract constraint
    any_of_constraint = AnyOf.get(BaseAttr(IntegerAttr), BaseAttr(AbstractTestAttr))
    assert isinstance(any_of_constraint, AnyOf)
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      scf.if %1 {
        %2 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
        irdl_interp.assert %2, "Expected integer"
      } else {
        %3 = irdl_interp.check_base %0 : !irdl_interp.attr, "test.abstract"
        irdl_interp.assert %3, "Expected test.abstract"
      }
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


def test_any_of_constraint_emit_with_var_constraint():
    """Test that AnyOf constraint with VarConstraint correctly passes EmitContext."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with (the value to verify)
    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    # Create a variable first (simulating Phase 2 setup)
    var_op = NewVarAttrOp()
    builder.insert(var_op)

    # Create context with the variable
    context = EmitContext(_attr_variables={"T": var_op.result})

    # Create AnyOf with VarConstraint and BaseAttr
    # VarConstraint("T", BaseAttr(IntegerAttr))
    var_constraint = VarConstraint("T", BaseAttr(IntegerAttr))
    any_of_constraint = AnyOf.get(var_constraint, BaseAttr(StringAttr))
    assert isinstance(any_of_constraint, AnyOf)
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context)

    # Expected IR
    expected_ir = """
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.new_var.attr
      %2 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      scf.if %2 {
        %3 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
        irdl_interp.assert %3, "Expected integer"
        %4 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %1 : !irdl_interp.variable.attr
        irdl_interp.assert %4, "Expected to unify with variable T"
      } else {
        %5 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
        scf.if %5 {
          %6 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
          irdl_interp.assert %6, "Expected string"
        } else {
          %7 = arith.constant false
          irdl_interp.assert %7, "Unexpected attribute"
        }
      }
    """
    expected_module = _parse_module(expected_ir)

    print(module)
    assert module.is_structurally_equivalent(expected_module)
