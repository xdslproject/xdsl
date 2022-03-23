from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.arith import *

test_prog = """
module() {
  %0 : !i1 = arith.constant() ["value" = 0 : !i1]
  %1 : !i1 = arith.andi(%0 : !i1, %0 : !i1)
}
"""


def test_main():
    ctx = MLContext()
    builtin = Builtin(ctx)
    arith = Arith(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    assert isinstance(module, ModuleOp)
    constant_op = module.ops[0]
    andi_op = module.ops[1]
    assert constant_op.results[0].uses == {Use(andi_op, 0), Use(andi_op, 1)}
    assert andi_op.results[0].uses == set()

    print("Done")
