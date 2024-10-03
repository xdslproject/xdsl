from xdsl.context import MLContext
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.test import Test
from xdsl.ir import Use
from xdsl.parser import Parser

test_prog = """
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""


def test_main():
    ctx = MLContext()
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
func.func @blocks() {
  "test.termop"() [^0, ^1] : () -> ()
^0:
  cf.br ^2
^1:
  cf.br ^2
^2:
  func.return
}
"""


def test_predecessor():
    ctx = MLContext()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)
    ctx.load_dialect(Func)
    ctx.load_dialect(Cf)

    parser = Parser(ctx, test_prog_blocks)
    func = parser.parse_op()

    func.verify()
    assert isinstance(func, FuncOp)

    block1, block2, block3, block4 = func.body.blocks

    assert block1.predecessors() == ()
    assert block2.predecessors() == (block1,)
    assert block3.predecessors() == (block1,)
    assert set(block4.predecessors()) == {block2, block3}
