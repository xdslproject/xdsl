from typing import Any
from unittest.mock import MagicMock

import pytest

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i32
from xdsl.ir import Block, SSAValue


def test_convert_indirect_call_raises():
    block = Block(arg_types=[i32])
    arg = block.args[0]
    op = llvm.CallOp("dummy_callee", arg, return_type=i32)

    # simulate indirect call
    op.callee = None

    builder = MagicMock()
    val_map: dict[SSAValue, Any] = {arg: MagicMock()}

    with pytest.raises(NotImplementedError, match="Indirect calls not yet implemented"):
        convert_op(op, builder, val_map)
