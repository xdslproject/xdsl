from xdsl.dialects.test import Test
from xdsl.dialects.builtin import ModuleOp, Builtin
from xdsl.ir import MLContext, Use
from xdsl.parser import Parser

test_prog = """
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""


def test_main():
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Test)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    assert isinstance(module, ModuleOp)

    op1, op2 = list(module.ops)
    assert op1.results[0].uses == {Use(op2, 0), Use(op2, 1)}
    assert op2.results[0].uses == set()

    print("Done")
