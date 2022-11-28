from xdsl.ir import MLContext, Use
from xdsl.dialects.builtin import ModuleOp, Builtin
from xdsl.parser import Parser
from xdsl.dialects.arith import Arith

test_prog = """
builtin.module() {
  %0 : !i1 = arith.constant() ["value" = 0 : !i1]
  %1 : !i1 = arith.andi(%0 : !i1, %0 : !i1)
}
"""


def test_main():
    ctx = MLContext()
    _ = Builtin(ctx)
    _ = Arith(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    assert isinstance(module, ModuleOp)
    constant_op = module.ops[0]
    andi_op = module.ops[1]
    assert constant_op.results[0].uses == {Use(andi_op, 0), Use(andi_op, 1)}
    assert andi_op.results[0].uses == set()

    print("Done")
