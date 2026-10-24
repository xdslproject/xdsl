import pytest

import xdsl.dialects.py as py
from xdsl.dialects.builtin import StringAttr


def test_custom_inits():
    c1 = py.PyConstOp.from_value(1)
    c2 = py.PyConstOp.from_value(2)
    b1 = py.PyBinOp(op=StringAttr("add"), lhs=c1.results[0], rhs=c2.results[0])
    assert b1.operands[0] is c1.results[0]
    assert b1.operands[1] is c2.results[0]
    cmp = py.PyCompareOp(op=StringAttr("lt"), lhs=c1.results[0], rhs=c2.results[0])
    assert cmp.operands[0] is c1.results[0]
    assert cmp.operands[1] is c2.results[0]


@pytest.mark.parametrize("value", [0, -(2**100), True, False, 1.5, "s"])
def test_const_from_value(value: int | bool | float | str):
    const = py.PyConstOp.from_value(value)
    assert const.value == value
    assert type(const.value) is type(value)


def test_call_keyword_names():
    c1 = py.PyConstOp.from_value(1)
    c2 = py.PyConstOp.from_value(2)
    args = (c1.results[0], c2.results[0])
    assert py.PyCallOp("f", args).keyword_names == ()
    assert py.PyCallOp("f", args, ("b",)).keyword_names == ("b",)
