import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects.test import Test, TestOp, TestTermOp
from xdsl.ir import Block
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


def test_operand_value():
    val1, val2 = TestOp.create(result_types=[i32, i32]).results
    op = TestOp.create(operands=[val1, val2])
    (use1,) = val1.uses
    (use2,) = val2.uses

    assert use1.value is val1
    assert use2.value is val2

    op.operands[0] = val2

    assert use1.value is val2
    assert use2.value is val2


@pytest.mark.parametrize("new_length", [0, 1, 2])
def test_operand_value_invalidated(new_length: int):
    values = TestOp.create(result_types=[i32, i32]).results
    op = TestOp.create(operands=values)
    uses = tuple(use for value in values for use in value.uses)

    op.operands = values[:new_length]

    for use in uses:
        with pytest.raises(
            AssertionError, match="Attempting to use Operand that has been replaced"
        ):
            _ = use.value

    for value in values[:new_length]:
        (use,) = value.uses
        assert use.value is value


def test_successor_value():
    block1, block2 = Block(), Block()
    op = TestTermOp.create(successors=[block1, block2])
    (use1,) = block1.uses
    (use2,) = block2.uses

    assert use1.value is block1
    assert use2.value is block2

    op.successors[0] = block2

    assert use1.value is block2
    assert use2.value is block2


@pytest.mark.parametrize("new_length", [0, 1, 2])
def test_successor_value_invalidated(new_length: int):
    blocks = (Block(), Block())
    op = TestTermOp.create(successors=blocks)
    uses = tuple(use for block in blocks for use in block.uses)

    op.successors = blocks[:new_length]

    for use in uses:
        with pytest.raises(
            AssertionError, match="Attempting to use Successor that has been replaced"
        ):
            _ = use.value

    for block in blocks[:new_length]:
        (use,) = block.uses
        assert use.value is block


def test_drop_all_references_invalidates_uses():
    (value,) = TestOp.create(result_types=[i32]).results
    block = Block()
    op = TestTermOp.create(operands=[value], successors=[block])
    (operand_use,) = value.uses
    (successor_use,) = block.uses

    op.drop_all_references()

    with pytest.raises(
        AssertionError, match="Attempting to use Operand that has been replaced"
    ):
        _ = operand_use.value
    with pytest.raises(
        AssertionError, match="Attempting to use Successor that has been replaced"
    ):
        _ = successor_use.value


def test_drop_all_references_clears_successor_uses():
    """Test that drop_all_references clears both operand and successor uses."""
    block1 = Block()
    block2 = Block()

    dummy_op = TestOp.create(result_types=[i32])
    val = dummy_op.results[0]

    op1 = TestTermOp.create(operands=[val], successors=[block1, block2])
    op2 = TestTermOp.create(successors=[block2])

    assert list(op1.successors) == [block1, block2]
    assert list(op2.successors) == [block2]
    assert len(op1.operands) == 1

    uses1 = list(block1.uses)
    uses2 = list(block2.uses)
    assert len(uses1) == 1
    assert len(uses2) == 2

    assert val.uses.get_length() == 1
    op1.drop_all_references()

    assert block1.uses.get_length() == 0
    assert block2.uses.get_length() == 1

    assert val.uses.get_length() == 0
    assert len(op1.successors) == 0
    assert op1.successor_uses == ()

    op2.drop_all_references()
    assert block2.uses.get_length() == 0
    assert len(op2.successors) == 0


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
