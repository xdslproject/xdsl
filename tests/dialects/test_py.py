import xdsl.dialects.py as py
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64


def test_custom_inits():
    c1 = py.PyConstOp(IntegerAttr(1, i64))
    c2 = py.PyConstOp(IntegerAttr(2, i64))
    b1 = py.PyBinOp(op=StringAttr("add"), lhs=c1.results[0], rhs=c2.results[0])
    assert b1.operands[0] is c1.results[0]
    assert b1.operands[1] is c2.results[0]
