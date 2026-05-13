import pytest

from xdsl.dialects.arith import AddiOp
from xdsl.dialects.builtin import i32
from xdsl.dialects.test import TestOp


def test_accessor_set():
    t = TestOp(result_types=(i32,))
    t2 = TestOp(result_types=(i32,))

    add = AddiOp(t, t)
    with pytest.raises(
        NotImplementedError, match="Setting operands via accessors is not implemented"
    ):
        add.lhs = t2  # pyright: ignore
