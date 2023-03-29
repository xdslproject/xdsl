from xdsl.ir import Block
from xdsl.dialects.arith import Addi, Subi, Muli, Constant
from xdsl.dialects.builtin import i32
from xdsl.dialects.cf import Branch, ConditionalBranch


def test_branch():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block.from_ops([a, b, c])
    br0 = Branch.get(block0)
    ops = list(br0.successors[0].iter_ops())

    assert br0.successors[0] is block0
    assert ops[0] is a
    assert ops[1] is b
    assert ops[2] is c


def test_condbranch():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)
    d = Subi.get(a, b)
    e = Muli.get(a, b)

    block0 = Block(arg_types=[i32])
    block1 = Block(arg_types=[i32])

    branch0 = ConditionalBranch.get(c, block0, [d], block1, [e])
    assert branch0.cond is c.results[0]
    assert branch0.then_arguments[0] is d.results[0]
    assert branch0.else_arguments[0] is e.results[0]
    assert branch0.successors[0] is block0
    assert branch0.successors[1] is block1
