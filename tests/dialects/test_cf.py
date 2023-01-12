from xdsl.ir import Block, OpResult
from xdsl.dialects.arith import Addi, Subi, Muli, Constant
from xdsl.dialects.builtin import i32
from xdsl.dialects.cf import Branch, ConditionalBranch


def test_branch():
    block0 = Block()
    _ = Branch.get(block0)


def test_condbranch():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)
    d = Subi.get(a, b)
    e = Muli.get(a, b)

    block0 = Block.from_arg_types([i32])
    block1 = Block.from_arg_types([i32])

    branch0 = ConditionalBranch.get(c, block0, [d], block1, [e])
    assert branch0.then is c.results[0]
    assert branch0.then_arguments[0] is d.results[0]
    assert branch0.else_arguments[0] is e.results[0]
    assert branch0.successors[0] is block0
    assert branch0.successors[1] is block1
