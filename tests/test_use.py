from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.test import Test, TestOp
from xdsl.ir import Use
from xdsl.parser import Parser

test_prog = """
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""


def test_main():
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    assert isinstance(module, ModuleOp)

    op1, op2 = list(module.ops)
    assert op1.results[0].uses == {Use(op2, 0), Use(op2, 1)}
    assert op2.results[0].uses == set()

    print("Done")


test_prog_blocks = """
"test.op"() ({
  "test.termop"() [^0, ^1] : () -> ()
^0:
  "test.termop"()[^2] : () -> ()
^1:
  "test.termop"()[^2] : () -> ()
^2:
  "test.termop"() : () -> ()
}) : () -> ()
"""


def test_predecessor():
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

    parser = Parser(ctx, test_prog_blocks)
    op = parser.parse_op()

    assert isinstance(op, TestOp)

    block1, block2, block3, block4 = op.regions[0].blocks

    assert block1.predecessors() == ()
    assert block2.predecessors() == (block1,)
    assert block3.predecessors() == (block1,)
    assert set(block4.predecessors()) == {block2, block3}
