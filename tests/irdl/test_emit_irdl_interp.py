"""
Tests for emit_irdl_interp functionality.

Uses is_structurally_equivalent to compare generated IR against expected IR.
"""

import pytest

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
    MessageConstraint,
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


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom base error: ",
    ],
)
def test_emit_base_attr_irdl_interp(message: str | None):
    """Test that BaseAttr constraint lowering generates correct IR with optional custom message."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    context = EmitContext()
    constraint = BaseAttr(IntegerAttr)

    constraint.emit_irdl_interp(constant_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      irdl_interp.assert %1, "{message or ""}Expected integer"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom any error",
    ],
)
def test_any_attr_emit_is_noop(message: str | None):
    """Test that AnyAttr.emit_irdl_interp is a no-op, even with custom message."""
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
    any_attr.emit_irdl_interp(constant_op.result, builder, context, message)

    # Count ops after - should be the same
    ops_after = len(list(block.ops))
    assert ops_before == ops_after


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom var error: ",
    ],
)
def test_var_constraint_emit_with_any_attr(message: str | None):
    """Test that VarConstraint with AnyAttr emits unify + assert with optional message."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    var_op = NewVarAttrOp()
    builder.insert(var_op)

    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    context = EmitContext(_attr_variables={"T": var_op.result})

    constraint = VarConstraint("T", AnyAttr())

    constraint.emit_irdl_interp(constant_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.new_var.attr
      %1 = irdl_interp.constant.attr 42 : index
      %2 = irdl_interp.unify.attr %1 : !irdl_interp.attr and %0 : !irdl_interp.variable.attr
      irdl_interp.assert %2, "{message or ""}Expected to unify with variable T"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom var error: ",
    ],
)
def test_var_constraint_emit_with_base_attr(message: str | None):
    """Test that VarConstraint with BaseAttr emits check + unify + assert with optional message."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    var_op = NewVarAttrOp()
    builder.insert(var_op)

    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    context = EmitContext(_attr_variables={"T": var_op.result})

    constraint = VarConstraint("T", BaseAttr(IntegerAttr))

    constraint.emit_irdl_interp(constant_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.new_var.attr
      %1 = irdl_interp.constant.attr 42 : index
      %2 = irdl_interp.check_base %1 : !irdl_interp.attr, "integer"
      irdl_interp.assert %2, "{message or ""}Expected integer"
      %3 = irdl_interp.unify.attr %1 : !irdl_interp.attr and %0 : !irdl_interp.variable.attr
      irdl_interp.assert %3, "{message or ""}Expected to unify with variable T"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom eq error: ",
    ],
)
def test_eq_attr_constraint_emit(message: str | None):
    """Test that EqAttrConstraint.emit_irdl_interp generates constant + unify + assert with optional message."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()
    expected_attr = IntegerAttr.from_index_int_value(42)
    constraint = EqAttrConstraint(expected_attr)

    constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.constant.attr 42 : index
      %2 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %1 : !irdl_interp.attr
      irdl_interp.assert %2, "{message or ""}Expected 42 : index"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom set error: ",
    ],
)
def test_attr_set_constraint_emit_single(message: str | None):
    """Test that AttrSetConstraint with single value generates false + constant + unify + OR + assert."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()
    expected_attr = IntegerAttr.from_index_int_value(42)
    set_constraint = AttrSetConstraint(frozenset([expected_attr]))
    set_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      %2 = irdl_interp.constant.attr 42 : index
      %3 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %2 : !irdl_interp.attr
      %4 = arith.ori %1, %3 : i1
      irdl_interp.assert %4, "{message or ""}Expected one of 42 : index"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom set error: ",
    ],
)
def test_attr_set_constraint_emit_multiple(message: str | None):
    """Test that AttrSetConstraint with multiple values generates false + constants + unify + OR + assert."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()
    attr1 = IntegerAttr.from_index_int_value(42)
    attr2 = IntegerAttr.from_index_int_value(43)
    set_constraint = AttrSetConstraint(frozenset([attr1, attr2]))
    set_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      %2 = irdl_interp.constant.attr 42 : index
      %3 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %2 : !irdl_interp.attr
      %4 = arith.ori %1, %3 : i1
      %5 = irdl_interp.constant.attr 43 : index
      %6 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %5 : !irdl_interp.attr
      %7 = arith.ori %4, %6 : i1
      irdl_interp.assert %7, "{message or ""}Expected one of 42 : index, 43 : index"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom set error: ",
    ],
)
def test_attr_set_constraint_emit_empty(message: str | None):
    """Test that AttrSetConstraint with empty set generates false constant + assert."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()
    set_constraint: AttrConstraint = AttrSetConstraint(frozenset({}))
    set_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      irdl_interp.assert %1, "{message or ""}Expected one of (empty set)"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom anyof error: ",
    ],
)
def test_any_of_constraint_emit(message: str | None):
    """Test that AnyOf constraint lowering generates scf.if dispatch."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()

    any_of_constraint = AnyOf.get(BaseAttr(IntegerAttr), BaseAttr(StringAttr))
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      scf.if %1 {{
        %2 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
        irdl_interp.assert %2, "{message or ""}Expected integer"
      }} else {{
        %3 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
        scf.if %3 {{
          %4 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
          irdl_interp.assert %4, "{message or ""}Expected string"
        }} else {{
          %5 = arith.constant false
          irdl_interp.assert %5, "{message or ""}Unexpected attribute"
        }}
      }}
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom anyof error: ",
    ],
)
def test_any_of_constraint_emit_empty(message: str | None):
    """Test that empty AnyOf constraint lowering generates immediate failure."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()

    any_of_constraint = AnyOf.get()
    assert isinstance(any_of_constraint, AnyOf)
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = arith.constant false
      irdl_interp.assert %1, "{message or ""}Unexpected attribute"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


class AbstractTestAttr(Attribute):
    name = "test.abstract"


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Custom anyof error: ",
    ],
)
def test_any_of_constraint_emit_with_abstract(message: str | None):
    """Test that AnyOf constraint with abstract constraint generates proper dispatch."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    context = EmitContext()

    any_of_constraint = AnyOf.get(BaseAttr(IntegerAttr), BaseAttr(AbstractTestAttr))
    assert isinstance(any_of_constraint, AnyOf)
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      scf.if %1 {{
        %2 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
        irdl_interp.assert %2, "{message or ""}Expected integer"
      }} else {{
        %3 = irdl_interp.check_base %0 : !irdl_interp.attr, "test.abstract"
        irdl_interp.assert %3, "{message or ""}Expected test.abstract"
      }}
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [None, "Custom anyof error: "],
)
def test_any_of_constraint_emit_with_var_constraint(message: str | None):
    """Test that AnyOf constraint with VarConstraint correctly passes EmitContext."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    value_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(value_op)

    var_op = NewVarAttrOp()
    builder.insert(var_op)

    context = EmitContext(_attr_variables={"T": var_op.result})

    var_constraint = VarConstraint("T", BaseAttr(IntegerAttr))
    any_of_constraint = AnyOf.get(var_constraint, BaseAttr(StringAttr))
    assert isinstance(any_of_constraint, AnyOf)
    any_of_constraint.emit_irdl_interp(value_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.new_var.attr
      %2 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      scf.if %2 {{
        %3 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
        irdl_interp.assert %3, "{message or ""}Expected integer"
        %4 = irdl_interp.unify.attr %0 : !irdl_interp.attr and %1 : !irdl_interp.variable.attr
        irdl_interp.assert %4, "{message or ""}Expected to unify with variable T"
      }} else {{
        %5 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
        scf.if %5 {{
          %6 = irdl_interp.check_base %0 : !irdl_interp.attr, "string"
          irdl_interp.assert %6, "{message or ""}Expected string"
        }} else {{
          %7 = arith.constant false
          irdl_interp.assert %7, "{message or ""}Unexpected attribute"
        }}
      }}
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)


@pytest.mark.parametrize(
    "message",
    [None, "Outer message: "],
)
def test_message_constraint_emit(message: str | None):
    """Test that MessageConstraint.emit_irdl_interp uses its own message, optionally prepended by outer message."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    context = EmitContext()

    message_constraint = MessageConstraint(
        BaseAttr(IntegerAttr), "Expected an integer attribute"
    )
    message_constraint.emit_irdl_interp(constant_op.result, builder, context, message)

    expected_ir = f"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      irdl_interp.assert %1, "{message or ""}Expected an integer attribute: Expected integer"
    """
    expected_module = _parse_module(expected_ir)

    print(module)

    assert module.is_structurally_equivalent(expected_module)


def test_message_constraint_nested():
    """Test that nested MessageConstraints properly chain messages."""
    module = ModuleOp([])
    block = module.body.block
    builder = Builder(InsertPoint.at_end(block))

    # Create a constant attribute value to test with
    constant_op = ConstantOp(IntegerAttr.from_index_int_value(42))
    builder.insert(constant_op)

    # Create empty context
    context = EmitContext()

    # Create double-nested MessageConstraint: Outer wraps Inner wraps BaseAttr
    inner_constraint = MessageConstraint(
        BaseAttr(IntegerAttr),
        "Inner message",
    )
    outer_constraint = MessageConstraint(
        inner_constraint,
        "Outer message",
    )
    outer_constraint.emit_irdl_interp(constant_op.result, builder, context)

    # Expected IR - should chain both messages:
    expected_ir = r"""
      %0 = irdl_interp.constant.attr 42 : index
      %1 = irdl_interp.check_base %0 : !irdl_interp.attr, "integer"
      irdl_interp.assert %1, "Outer message: Inner message: Expected integer"
    """
    expected_module = _parse_module(expected_ir)

    assert module.is_structurally_equivalent(expected_module)
