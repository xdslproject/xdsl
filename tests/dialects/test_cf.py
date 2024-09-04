from xdsl.dialects.arith import Addi, Constant, Muli, Subi
from xdsl.dialects.builtin import StringAttr, i1, i32
from xdsl.dialects.cf import Assert, Branch, ConditionalBranch
from xdsl.ir import Block


def test_assert():
    a = Constant.from_int_and_width(1, i1)
    b = Constant.from_int_and_width(1, i1)
    c = Assert(a, "a")
    d = Assert(b, StringAttr("b"))

    assert c.arg is a.result
    assert d.arg is b.result
    assert c.attributes["msg"] == StringAttr("a")
    assert d.attributes["msg"] == StringAttr("b")


def test_branch():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi(a, b)

    block0 = Block([a, b, c])
    br0 = Branch(block0)
    ops = list(br0.successors[0].ops)

    assert br0.successor is block0
    assert ops[0] is a
    assert ops[1] is b
    assert ops[2] is c


def test_condbranch():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi(a, b)
    d = Subi(a, b)
    e = Muli(a, b)

    block0 = Block(arg_types=[i32])
    block1 = Block(arg_types=[i32])

    branch0 = ConditionalBranch(c, block0, [d], block1, [e])
    assert branch0.cond is c.results[0]
    assert branch0.then_arguments[0] is d.results[0]
    assert branch0.else_arguments[0] is e.results[0]
    assert branch0.then_block is block0
    assert branch0.else_block is block1
