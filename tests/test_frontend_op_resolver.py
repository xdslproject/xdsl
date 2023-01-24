from __future__ import annotations

import pytest
import xdsl.frontend.dialects.builtin as builtin

from xdsl.frontend.exception import FrontendProgramException
from xdsl.frontend.op_resolver import OpResolver


def test_raises_exception_on_unknown_op():
    with pytest.raises(FrontendProgramException) as err:
        op = OpResolver.resolve_op("xdsl.frontend.dialects.arith", "unknown")
    assert err.value.msg == "Internal failure: operation 'unknown' does not exist in module 'xdsl.frontend.dialects.arith'."


def test_raises_exception_on_unknown_overload():
    with pytest.raises(FrontendProgramException) as err:
        op = OpResolver.resolve_op_overload("__unknown__", builtin._Integer)
    assert err.value.msg == "Internal failure: '_Integer' does not overload '__unknown__'."
