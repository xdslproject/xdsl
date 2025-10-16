from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.test import TestOp
from xdsl.utils.op_selector import OpSelector


def test_op_selector():
    a = TestOp()
    b = TestOp()
    c = TestOp()

    module = ModuleOp([a, b, c])

    assert OpSelector(0, "builtin.module").get_op(module) is module
    assert OpSelector(1, "test.op").get_op(module) is a
    assert OpSelector(2, "test.op").get_op(module) is b
    assert OpSelector(3, "test.op").get_op(module) is c

    import pytest

    with pytest.raises(IndexError, match="Matching index 4 out of range."):
        OpSelector(4, "test.op").get_op(module)

    with pytest.raises(
        ValueError, match="Unexpected op builtin.module at index 0, expected test.op."
    ):
        OpSelector(0, "test.op").get_op(module)
    with pytest.raises(
        ValueError, match="Unexpected op test.op at index 1, expected builtin.module."
    ):
        OpSelector(1, "builtin.module").get_op(module)
