# do not include 'from __future__ import annotations' in this file
from typing import ClassVar

import pytest

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr, i32
from xdsl.dialects.irdl import (
    AllOfOp,
    AnyOfOp,
    AnyOp,
    AttributeOp,
    AttributeType,
    BaseOp,
    DialectOp,
    IsOp,
    OperationOp,
    ParametricOp,
    TypeOp,
)
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition
from xdsl.utils.exceptions import PyRDLOpDefinitionError
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("op_type", [DialectOp, TypeOp, AttributeOp, OperationOp])
def test_named_region_op_init(
    op_type: type[DialectOp | TypeOp | AttributeOp | OperationOp],
):
    """
    Test __init__ of DialectOp, TypeOp, AttributeOp, OperationOp.
    """
    op = op_type("cmath", Region(Block()))
    op2 = op_type(StringAttr("cmath"), Region(Block()))
    op3 = op_type.create(
        attributes={"sym_name": StringAttr("cmath")}, regions=[Region(Block())]
    )

    assert op.is_structurally_equivalent(op2)
    assert op2.is_structurally_equivalent(op3)

    assert op.sym_name == StringAttr("cmath")
    assert len(op.body.blocks) == 1


def test_is_init():
    """Test __init__ of IsOp."""
    op = IsOp(i32)
    op2 = IsOp.create(attributes={"expected": i32}, result_types=[AttributeType()])

    assert op.is_structurally_equivalent(op2)

    assert op.expected == i32
    assert op.output.type == AttributeType()


def test_base_init():
    """Test __init__ of BaseOp."""
    base_op_ref = BaseOp(SymbolRefAttr("integer"))
    base_op_ref2 = BaseOp.create(
        attributes={"base_ref": SymbolRefAttr("integer")},
        result_types=[AttributeType()],
    )
    assert base_op_ref.is_structurally_equivalent(base_op_ref2)

    base_op_name = BaseOp(StringAttr("integer"))
    base_op_name2 = BaseOp("integer")
    base_op_name3 = BaseOp.create(
        attributes={"base_name": StringAttr("integer")},
        result_types=[AttributeType()],
    )
    assert base_op_name.is_structurally_equivalent(base_op_name2)
    assert base_op_name2.is_structurally_equivalent(base_op_name3)


def test_parametric_init():
    """Test __init__ of ParametricOp."""
    val1 = create_ssa_value(AttributeType())
    val2 = create_ssa_value(AttributeType())

    op = ParametricOp("complex", [val1, val2])
    op2 = ParametricOp(StringAttr("complex"), [val1, val2])
    op3 = ParametricOp(SymbolRefAttr("complex"), [val1, val2])
    op4 = ParametricOp.create(
        attributes={"base_type": SymbolRefAttr("complex")},
        operands=[val1, val2],
        result_types=[AttributeType()],
    )

    assert op.is_structurally_equivalent(op2)
    assert op2.is_structurally_equivalent(op3)
    assert op3.is_structurally_equivalent(op4)

    assert op.base_type == SymbolRefAttr("complex")
    assert op.args == (val1, val2)
    assert op.output.type == AttributeType()


def test_any_init():
    """Test __init__ of AnyOp."""
    op = AnyOp()
    op2 = AnyOp.create(result_types=[AttributeType()])

    assert op.is_structurally_equivalent(op2)
    assert op.output.type == AttributeType()


@pytest.mark.parametrize("op_type", [AllOfOp, AnyOfOp])
def test_any_all_of_init(op_type: type[AllOfOp | AnyOfOp]):
    """Test __init__ of AnyOf and AllOf."""
    val1 = create_ssa_value(AttributeType())
    val2 = create_ssa_value(AttributeType())
    op = op_type((val1, val2))
    op2 = op_type.create(operands=[val1, val2], result_types=[AttributeType()])

    assert op.is_structurally_equivalent(op2)

    assert op.args == (val1, val2)
    assert op.output.type == AttributeType()


@pytest.mark.parametrize("op_type", [OperationOp, TypeOp, AttributeOp])
def test_qualified_name(op_type: type[OperationOp | TypeOp | AttributeOp]):
    """Test qualified_name property of OperationOp, TypeOp, AttributeOp."""
    op = op_type.create(
        attributes={"sym_name": StringAttr("myname")}, regions=[Region(Block())]
    )
    dialect = DialectOp("mydialect", Region(Block([op])))
    dialect.verify()
    assert op.qualified_name == "mydialect.myname"


class MyOpWithClassVar(IRDLOperation):
    name = "test.has_class_var"

    VAR: ClassVar[str] = "hello_world"


class MyOpWithClassVarInvalid(IRDLOperation):
    name = "test.has_class_var"

    var: ClassVar[str] = "hello_world"


def test_class_var_on_op():
    irdl_op_definition(MyOpWithClassVar)


def test_class_var_on_op_invalid():
    with pytest.raises(
        PyRDLOpDefinitionError, match='Invalid ClassVar name "var", must be uppercase'
    ):
        irdl_op_definition(MyOpWithClassVarInvalid)


class MySuperWithClassVarDef(IRDLOperation):
    name = "test.super_has_class_var"

    VAR: ClassVar[str]


class MySubWithClassVarOverload(MySuperWithClassVarDef):
    name = "test.super_has_class_var"

    VAR: ClassVar[str] = "hello_world"


def test_class_var_on_super():
    irdl_op_definition(MySuperWithClassVarDef)
    irdl_op_definition(MySubWithClassVarOverload)


@pytest.mark.parametrize(
    "op_name, class_name",
    [
        ("my_operation", "MyOperationOp"),
        ("AlreadyCamelCase", "AlreadyCamelCaseOp"),
        ("nested.name", "NestedNameOp"),
    ],
)
def test_get_py_class_name(op_name: str, class_name: str):
    op = OperationOp(op_name, Region(Block()))
    assert op.get_py_class_name() == class_name
