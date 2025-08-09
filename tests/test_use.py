from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.test import Test, TestOp
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
    assert set((use.operation, use.index) for use in op1.results[0].uses) == {
        (op2, 0),
        (op2, 1),
    }
    assert set(op2.results[0].uses) == set()


def test_uses_methods():
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Test)

    test_prog = """
    "builtin.module"() ({
      %0 = "test.op"() : () -> i32
      %1 = "test.op"(%0, %0) : (i32, i32) -> i32
      %2 = "test.op"(%1) : (i32) -> i32
    }) : () -> ()
    """

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    assert isinstance(module, ModuleOp)

    op0, op1, op2 = list(module.ops)
    res0, res1, res2 = op0.results[0], op1.results[0], op2.results[0]
    assert not res0.has_one_use()
    assert res1.has_one_use()
    assert not res2.has_one_use()

    assert res0.has_more_than_one_use()
    assert not res1.has_more_than_one_use()
    assert not res2.has_more_than_one_use()

    assert res0.get_unique_use() is None
    assert (use := res1.get_unique_use()) is not None
    assert use.operation == op2
    assert use.index == 0
    assert res2.get_unique_use() is None

    assert res0.get_user_of_unique_use() is None
    assert res1.get_user_of_unique_use() == op2
    assert res2.get_user_of_unique_use() is None


test_prog_blocks = """
"test.op"() ({
  "test.termop"() [^bb0, ^bb1] : () -> ()
^bb0:
  "test.termop"()[^bb2] : () -> ()
^bb1:
  "test.termop"()[^bb2] : () -> ()
^bb2:
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
