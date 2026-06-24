import pytest

from xdsl.dialects import test
from xdsl.utils.worklist import Worklist


def test_worklist_push_pop():
    """Test push/pop operations on the worklist."""
    op1 = test.TestOp()
    op2 = test.TestOp()
    op3 = test.TestOp()
    op4 = test.TestOp()

    worklist = Worklist()

    worklist.push(op1)
    worklist.push(op3)
    worklist.push(op2)
    worklist.push(op4)

    assert worklist.pop() is op4
    assert worklist.pop() is op2
    worklist.push(op4)
    assert worklist.pop() is op4
    assert worklist.pop() is op3
    assert worklist.pop() is op1
    assert not worklist
    with pytest.raises(IndexError, match="pop from empty worklist"):
        worklist.pop()


def test_worklist_push_already_inserted():
    """Test push of an already inserted operation."""

    op1 = test.TestOp()
    op2 = test.TestOp()

    worklist = Worklist()
    worklist.push(op1)
    worklist.push(op2)

    worklist.push(op1)

    assert worklist.pop() is op2
    assert worklist.pop() is op1
    assert not worklist
    with pytest.raises(IndexError, match="pop from empty worklist"):
        worklist.pop()


def test_worklist_remove():
    """Test remove operation."""
    op1 = test.TestOp()
    op2 = test.TestOp()
    op3 = test.TestOp()
    op4 = test.TestOp()

    worklist = Worklist()

    worklist.push(op1)
    worklist.push(op3)
    worklist.push(op2)
    worklist.push(op4)

    worklist.remove(op1)
    worklist.remove(op2)
    worklist.remove(op2)

    assert worklist.pop() is op4
    assert worklist.pop() is op3
    assert not worklist
    with pytest.raises(IndexError, match="pop from empty worklist"):
        worklist.pop()


def test_worklist_non_default():
    worklist = Worklist[int]()

    worklist.push(1)
    worklist.push(2)
    assert worklist.pop() == 2
    assert worklist.pop() == 1
    assert not worklist


def test_worklist_with_none():
    worklist = Worklist[int | None]()

    worklist.push(None)
    worklist.push(1)
    assert worklist.pop() == 1
    assert worklist.pop() is None

    with pytest.raises(IndexError, match="pop from empty worklist"):
        worklist.pop()

    worklist.push(1)
    worklist.push(None)
    worklist.push(2)
    worklist.remove(None)

    assert worklist.pop() == 2
    assert worklist.pop() == 1
    assert not worklist
