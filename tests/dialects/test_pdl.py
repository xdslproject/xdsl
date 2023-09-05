import pytest

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, i32, i64
from xdsl.ir import Block
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue

type_type = pdl.TypeType()
attribute_type = pdl.AttributeType()
value_type = pdl.ValueType()
operation_type = pdl.OperationType()

block = Block(
    arg_types=[
        type_type,
        attribute_type,
        value_type,
        operation_type,
    ]
)

type_val, attr_val, val_val, op_val = block.args


def test_build_anc():
    anc = pdl.ApplyNativeConstraintOp("anc", [type_val])

    assert anc.attributes["name"] == StringAttr("anc")
    assert anc.args == (type_val,)


def test_build_anr():
    anr = pdl.ApplyNativeRewriteOp("anr", [type_val], [attribute_type])

    assert anr.attributes["name"] == StringAttr("anr")
    assert anr.args == (type_val,)
    assert len(anr.results) == 1
    assert [r.type for r in anr.results] == [attribute_type]


def test_build_rewrite():
    r = pdl.RewriteOp(
        name="r", root=None, external_args=[type_val, attr_val], body=None
    )

    assert r.attributes["name"] == StringAttr("r")
    assert r.external_args == (type_val, attr_val)
    assert len(r.results) == 0
    assert r.body is None

    r1 = pdl.RewriteOp(name="r", root=None, external_args=[type_val, attr_val])

    assert r1.body is not None


def test_build_operation_replace():
    operation = pdl.OperationOp(
        op_name="operation",
        attribute_value_names=ArrayAttr([StringAttr("name")]),
        operand_values=[val_val],
        attribute_values=[attr_val],
        type_values=[type_val],
    )

    assert operation.opName == StringAttr("operation")
    assert operation.attributeValueNames == ArrayAttr([StringAttr("name")])
    assert operation.operand_values == (val_val,)
    assert operation.attribute_values == (attr_val,)
    assert operation.type_values == (type_val,)

    replace = pdl.ReplaceOp(op_value=op_val, repl_operation=operation.results[0])
    replace.verify()

    assert replace.op_value == op_val
    assert replace.repl_operation == operation.results[0]
    assert replace.repl_values == ()

    replace = pdl.ReplaceOp(op_value=op_val, repl_values=[val_val])
    replace.verify()

    assert replace.op_value == op_val
    assert replace.repl_operation is None
    assert replace.repl_values == (val_val,)

    with pytest.raises(VerifyException):
        replace = pdl.ReplaceOp(op_value=op_val)
        replace.verify()

    with pytest.raises(VerifyException):
        replace = pdl.ReplaceOp(
            op_value=op_val, repl_operation=operation.results[0], repl_values=[val_val]
        )
        replace.verify()


def test_build_result():
    res = pdl.ResultOp(IntegerAttr.from_int_and_width(1, 32), parent=op_val)

    assert res.index == IntegerAttr.from_int_and_width(1, 32)
    assert res.parent_ == op_val


def test_build_resultS():
    res = pdl.ResultsOp(op_val)

    assert res.parent_ == op_val
    assert res.index is None
    assert res.val.type == pdl.RangeType(pdl.ValueType())


def test_build_results_with_index():
    res = pdl.ResultsOp(op_val, IntegerAttr.from_int_and_width(1, 32))

    assert res.parent_ == op_val
    assert res.index == IntegerAttr.from_int_and_width(1, 32)
    assert res.val.type == pdl.RangeType(pdl.ValueType())


def test_build_results_with_index_and_type():
    res = pdl.ResultsOp(op_val, IntegerAttr.from_int_and_width(1, 32), pdl.ValueType())

    assert res.parent_ == op_val
    assert res.index == IntegerAttr.from_int_and_width(1, 32)
    assert res.val.type == pdl.ValueType()


def test_build_type():
    pdl_type = pdl.TypeOp()
    assert pdl_type.constantType is None

    pdl_type = pdl.TypeOp(i32)
    assert pdl_type.constantType == i32


def test_build_types():
    pdl_type = pdl.TypesOp()
    assert pdl_type.constantTypes is None

    pdl_type = pdl.TypesOp((i32, i64, i32))
    assert pdl_type.constantTypes == ArrayAttr((i32, i64, i32))


def test_build_operand():
    operand = pdl.OperandOp(val_val)

    assert operand.value_type == val_val


def test_range():
    val1 = TestSSAValue(pdl.ValueType())
    val2 = TestSSAValue(pdl.RangeType(pdl.ValueType()))
    val3 = TestSSAValue(pdl.ValueType())

    range_op = pdl.RangeOp((val1, val2, val3))

    assert range_op.arguments == (val1, val2, val3)
    assert range_op.result.type == pdl.RangeType(pdl.ValueType())


def test_empty_range():
    return_type = pdl.RangeType(pdl.ValueType())
    empty_range = pdl.RangeOp((), return_type)

    assert len(empty_range.arguments) == 0
    assert empty_range.result.type == return_type


def test_range_cannot_infer():
    with pytest.raises(ValueError):
        pdl.RangeOp(())  # Cannot infer return type
