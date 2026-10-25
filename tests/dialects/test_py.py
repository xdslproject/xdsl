import pytest

import xdsl.dialects.py as py
from xdsl.dialects.builtin import FunctionType
from xdsl.ir import Block, Region


def test_custom_inits():
    c1 = py.PyConstOp.from_value(1)
    c2 = py.PyConstOp.from_value(2)
    b1 = py.PyBinOp(op=py.Operator.ADD, lhs=c1.results[0], rhs=c2.results[0])
    assert b1.operands[0] is c1.results[0]
    assert b1.operands[1] is c2.results[0]
    cmp = py.PyCompareOp(op=py.CmpOp.LT, lhs=c1.results[0], rhs=c2.results[0])
    assert cmp.operands[0] is c1.results[0]
    assert cmp.operands[1] is c2.results[0]


@pytest.mark.parametrize("value", [0, -(2**100), True, False, 1.5, "s", None])
def test_const_from_value(value: int | bool | float | str | None):
    const = py.PyConstOp.from_value(value)
    assert const.value == value
    assert type(const.value) is type(value)


def test_call_keyword_names():
    c1 = py.PyConstOp.from_value(1)
    c2 = py.PyConstOp.from_value(2)
    args = (c1.results[0], c2.results[0])
    assert py.PyCallOp("f", args).keyword_names == ()
    assert py.PyCallOp("f", args, ("b",)).keyword_names == ("b",)


def test_build_tuple_init():
    elements = (py.PyConstOp.from_value(1).res, py.PyConstOp.from_value("s").res)
    assert tuple(py.PyBuildTupleOp(elements).elements) == elements


def test_object_annotation():
    assert py.PyObjectType() == py.PyObjectType("")
    assert py.PyObjectType("int").annotation.data == "int"


def test_unpack_init():
    unpack = py.PyUnpackOp(py.PyConstOp.from_value(1).res, [py.PyObjectType()] * 3)
    assert unpack.res.types == (py.PyObjectType(),) * 3


def test_func_init():
    int_type = py.PyObjectType("int")
    body = Region(Block(arg_types=[int_type]))
    zero = py.PyConstOp.from_value(0)
    body.block.add_ops([zero, py.PyReturnOp(zero.res)])
    func = py.PyFuncOp("f", FunctionType.from_lists([int_type], [int_type]), body)
    func.verify()
    assert func.func_name.data == "f"
    assert func.body.block.args[0].type == int_type
