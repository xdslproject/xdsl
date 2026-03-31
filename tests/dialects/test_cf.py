from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp, SubiOp
from xdsl.dialects.builtin import StringAttr, i1, i32
from xdsl.dialects.cf import AssertOp, BranchOp, ConditionalBranchOp
from xdsl.ir import Block


def test_assert():
    a = ConstantOp.from_int_and_width(1, i1)
    b = ConstantOp.from_int_and_width(1, i1)
    c = AssertOp(a, "a")
    d = AssertOp(b, StringAttr("b"))

    assert c.arg is a.result
    assert d.arg is b.result
    assert c.properties["msg"] == StringAttr("a")
    assert d.properties["msg"] == StringAttr("b")


def test_branch():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    # Operation to add these constants
    c = AddiOp(a, b)

    block0 = Block([a, b, c])
    br0 = BranchOp(block0)
    ops = list(br0.successors[0].ops)

    assert br0.successor is block0
    assert ops[0] is a
    assert ops[1] is b
    assert ops[2] is c


def test_condbranch():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    # Operation to add these constants
    c = AddiOp(a, b)
    d = SubiOp(a, b)
    e = MuliOp(a, b)

    block0 = Block(arg_types=[i32])
    block1 = Block(arg_types=[i32])

    branch0 = ConditionalBranchOp(c, block0, [d], block1, [e])
    assert branch0.cond is c.results[0]
    assert branch0.then_arguments[0] is d.results[0]
    assert branch0.else_arguments[0] is e.results[0]
    assert branch0.then_block is block0
    assert branch0.else_block is block1
