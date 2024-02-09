import pytest

from xdsl.dialects import test
from xdsl.ir import Block

from xdsl.transforms.utils import (
    find_corresponding_store,
    get_operation_at_index,
    is_loop_dependent,
)


def test_get_operation_at_index():
    bb0 = Block()

    with pytest.raises(ValueError):
        get_operation_at_index(bb0, 0)

    with pytest.raises(ValueError):
        get_operation_at_index(bb0, 1)

    with pytest.raises(ValueError):
        get_operation_at_index(bb0, -1)

    op0 = test.TestOp()
    bb1 = Block([op0])

    assert op0 == get_operation_at_index(bb1, 0)

    with pytest.raises(ValueError):
        get_operation_at_index(bb1, 1)

    with pytest.raises(ValueError):
        get_operation_at_index(bb1, -1)

    op1 = test.TestOp()
    op2 = test.TestOp()
    bb2 = Block([op1, op2])

    assert op1 == get_operation_at_index(bb2, 0)
    assert op2 == get_operation_at_index(bb2, 1)

    with pytest.raises(ValueError):
        get_operation_at_index(bb2, 2)
