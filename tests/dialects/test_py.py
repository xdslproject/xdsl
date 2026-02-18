import xdsl.dialects.py as py
from xdsl.dialects.builtin import IntegerAttr, StringAttr


def test_custom_inits():
    c1 = py.PyConstOp(IntegerAttr.from_int_and_width(1, 64))
    c2 = py.PyConstOp(IntegerAttr.from_int_and_width(2, 64))
    b1 = py.PyBinOp(op=StringAttr("add"), lhs=c1.results[0], rhs=c2.results[0])
    assert b1.operands[0] is c1.results[0]
    assert b1.operands[1] is c2.results[0]
