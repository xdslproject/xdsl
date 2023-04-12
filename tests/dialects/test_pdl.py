import pytest
from xdsl.ir import Block

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

type_type = pdl.TypeType()
attribute_type = pdl.AttributeType()
value_type = pdl.ValueType()
operation_type = pdl.OperationType()

block = Block(arg_types=[
    type_type,
    attribute_type,
    value_type,
    operation_type,
])

type_val, attr_val, val_val, op_val = block.args


def test_build_anc():
    anc = pdl.ApplyNativeConstraintOp.get('anc', [type_val])

    assert anc.attributes['name'] == StringAttr('anc')
    assert anc.args == (type_val, )


def test_build_anr():
    anr = pdl.ApplyNativeRewriteOp.get('anr', [type_val], [attribute_type])

    assert anr.attributes['name'] == StringAttr('anr')
    assert anr.args == (type_val, )
    assert len(anr.results) == 1
    assert [r.typ for r in anr.results] == [attribute_type]


def test_build_rewrite():
    r = pdl.RewriteOp.get(StringAttr('r'),
                          root=None,
                          external_args=[type_val, attr_val],
                          body=None)

    assert r.attributes['name'] == StringAttr('r')
    assert r.externalArgs == (type_val, attr_val)
    assert len(r.results) == 0


def test_build_operation_replace():
    operation = pdl.OperationOp.get(opName=StringAttr('operation'),
                                    attributeValueNames=ArrayAttr(
                                        [StringAttr('name')]),
                                    operandValues=[val_val],
                                    attributeValues=[attr_val],
                                    typeValues=[type_val])

    assert operation.opName == StringAttr('operation')
    assert operation.attributeValueNames == ArrayAttr([StringAttr('name')])
    assert operation.operandValues == (val_val, )
    assert operation.attributeValues == (attr_val, )
    assert operation.typeValues == (type_val, )

    replace = pdl.ReplaceOp.get(opValue=op_val,
                                replOperation=operation.results[0])
    replace.verify()

    assert replace.opValue == op_val
    assert replace.replOperation == operation.results[0]
    assert replace.replValues == ()

    replace = pdl.ReplaceOp.get(opValue=op_val, replValues=[val_val])
    replace.verify()

    assert replace.opValue == op_val
    assert replace.replOperation == None
    assert replace.replValues == (val_val, )

    with pytest.raises(VerifyException):
        replace = pdl.ReplaceOp.get(opValue=op_val)
        replace.verify()

    with pytest.raises(VerifyException):
        replace = pdl.ReplaceOp.get(opValue=op_val,
                                    replOperation=operation.results[0],
                                    replValues=[val_val])
        replace.verify()


def test_build_result():
    res = pdl.ResultOp.get(IntegerAttr.from_int_and_width(1, 32),
                           parent=op_val)

    assert res.index == IntegerAttr.from_int_and_width(1, 32)
    assert res.parent_ == op_val


def test_build_operand():
    operand = pdl.OperandOp.get(val_val)

    assert operand.valueType == val_val
