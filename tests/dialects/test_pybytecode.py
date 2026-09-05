import xdsl.dialects.pybytecode as pyb
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64


def test_custom_inits():
    c1 = pyb.PyConstOp(IntegerAttr(1, i64))
    c2 = pyb.PyConstOp(IntegerAttr(2, i64))
    b1 = pyb.PyBinOp(op=StringAttr("add"), lhs=c1.results[0], rhs=c2.results[0])
    assert b1.operands[0] is c1.results[0]
    assert b1.operands[1] is c2.results[0]
