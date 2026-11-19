import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp, i32
from xdsl.dialects.test import Test, TestOp, TestTermOp
from xdsl.ir import Block
from xdsl.parser import Parser
from xdsl.utils.test_value import create_ssa_value

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


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("same_value", [False, True])
@pytest.mark.parametrize("via_setter", [False, True])
def test_operand_replace_use(index: int, same_value: bool, via_setter: bool):
    old_value = create_ssa_value(i32)
    new_value = old_value if same_value else create_ssa_value(i32)
    op = TestOp.create(operands=[old_value] * 3 + [new_value])
    old_uses = tuple(old_value.uses)
    new_uses = tuple(new_value.uses)
    use = next(use for use in old_uses if use.index == index)

    if via_setter:
        op.operands[index] = new_value
    else:
        new_value.replace_use(use)

    assert use.operation is op
    assert use.index == index
    assert use.value is new_value
    assert tuple(op.operands) == tuple(
        new_value if i == index or i == 3 else old_value for i in range(4)
    )
    if same_value:
        assert tuple(old_value.uses) == old_uses
    else:
        assert tuple(old_value.uses) == tuple(u for u in old_uses if u is not use)
        assert tuple(new_value.uses) == (use, *new_uses)


@pytest.mark.parametrize("new_length", [0, 1, 2])
def test_operand_value_invalidated(new_length: int):
    values = TestOp.create(result_types=[i32, i32]).results
    op = TestOp.create(operands=values)
    uses = tuple(use for value in values for use in value.uses)

    op.set_operands(values[:new_length])

    for use in uses:
        with pytest.raises(
            AssertionError, match="Attempting to use Operand that has been replaced"
        ):
            _ = use.value
        with pytest.raises(
            AssertionError, match="Attempting to use Operand that has been replaced"
        ):
            values[0].replace_use(use)

    assert tuple(op.operands) == values[:new_length]

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


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("same_value", [False, True])
@pytest.mark.parametrize("via_setter", [False, True])
def test_successor_replace_use(index: int, same_value: bool, via_setter: bool):
    old_block = Block()
    new_block = old_block if same_value else Block()
    op = TestTermOp.create(successors=[old_block] * 3 + [new_block])
    old_uses = tuple(old_block.uses)
    new_uses = tuple(new_block.uses)
    use = next(use for use in old_uses if use.index == index)

    if via_setter:
        op.successors[index] = new_block
    else:
        new_block.replace_use(use)

    assert use.operation is op
    assert use.index == index
    assert use.value is new_block
    assert tuple(op.successors) == tuple(
        new_block if i == index or i == 3 else old_block for i in range(4)
    )
    if same_value:
        assert tuple(old_block.uses) == old_uses
    else:
        assert tuple(old_block.uses) == tuple(u for u in old_uses if u is not use)
        assert tuple(new_block.uses) == (use, *new_uses)


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
        with pytest.raises(
            AssertionError, match="Attempting to use Successor that has been replaced"
        ):
            blocks[0].replace_use(use)

    assert tuple(op.successors) == blocks[:new_length]

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
    with pytest.raises(
        AssertionError, match="Attempting to use Operand that has been replaced"
    ):
        value.replace_use(operand_use)
    with pytest.raises(
        AssertionError, match="Attempting to use Successor that has been replaced"
    ):
        block.replace_use(successor_use)


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
