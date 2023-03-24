import pytest
from xdsl.ir import Block

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

type_type = pdl.TypeType()
attribute_type = pdl.AttributeType()
value_type = pdl.ValueType()
operation_type = pdl.OperationType()


def test_build_ops():

    block = Block.from_arg_types([
        type_type,
        attribute_type,
        value_type,
        operation_type,
    ])
    arg0, arg1, arg2, arg3 = block.args

    anc = pdl.ApplyNativeConstraintOp.get('anc', [arg0])

    assert anc.attributes['name'] == StringAttr('anc')
    assert anc.args == (arg0, )

    anr = pdl.ApplyNativeRewriteOp.get('anr', [arg0], [attribute_type])

    assert anr.attributes['name'] == StringAttr('anr')
    assert anr.args == (arg0, )
    assert len(anr.results) == 1
    assert [r.typ for r in anr.results] == [attribute_type]

    r = pdl.RewriteOp.get(StringAttr('r'),
                          root=None,
                          external_args=[arg0, arg1],
                          body=None)

    assert r.attributes['name'] == StringAttr('r')
    assert r.externalArgs == (arg0, arg1)
    assert len(r.results) == 0

    operation = pdl.OperationOp.get(opName=StringAttr('operation'),
                                    attributeValueNames=ArrayAttr(
                                        [StringAttr('name')]),
                                    operandValues=[arg2],
                                    attributeValues=[arg1],
                                    typeValues=[arg0])

    assert operation.opName == StringAttr('operation')
    assert operation.attributeValueNames == ArrayAttr([StringAttr('name')])
    assert operation.operandValues == (arg2, )
    assert operation.attributeValues == (arg1, )
    assert operation.typeValues == (arg0, )

    res = pdl.ResultOp.get(IntegerAttr.from_int_and_width(1, 32), parent=arg3)

    assert res.index == IntegerAttr.from_int_and_width(1, 32)
    assert res.parent_ == arg3

    operand = pdl.OperandOp.get(arg2)

    assert operand.valueType == arg2

    replace = pdl.ReplaceOp.get(opValue=arg3,
                                replOperation=operation.results[0])
    replace.verify()

    assert replace.opValue == arg3
    assert replace.replOperation == operation.results[0]
    assert replace.replValues == ()

    replace = pdl.ReplaceOp.get(opValue=arg3, replValues=[arg2])
    replace.verify()

    assert replace.opValue == arg3
    assert replace.replOperation == None
    assert replace.replValues == (arg2, )

    with pytest.raises(VerifyException):
        replace = pdl.ReplaceOp.get(opValue=arg3)
        replace.verify()

    with pytest.raises(VerifyException):
        replace = pdl.ReplaceOp.get(opValue=arg3,
                                    replOperation=operation.results[0],
                                    replValues=[arg2])
        replace.verify()
