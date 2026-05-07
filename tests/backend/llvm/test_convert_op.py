from typing import Any
from unittest.mock import MagicMock

import pytest

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i32
from xdsl.dialects.utils import FastMathFlag
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


def test_call_intrinsic_op_bundle_raises():
    block = Block(arg_types=[i32])
    arg = block.args[0]
    op = llvm.CallIntrinsicOp("llvm.donothing", [arg], [], op_bundle_operands=[arg])

    with pytest.raises(NotImplementedError, match="Operand bundles not supported"):
        convert_op(op, MagicMock(), {arg: MagicMock()})


def test_call_intrinsic_fastmath_raises():
    block = Block(arg_types=[i32])
    arg = block.args[0]
    op = llvm.CallIntrinsicOp("llvm.donothing", [arg], [])
    op.properties["fastmathFlags"] = llvm.FastMathAttr([FastMathFlag.NO_NANS])

    with pytest.raises(NotImplementedError, match="Fast-math flags not supported"):
        convert_op(op, MagicMock(), {arg: MagicMock()})


def test_fma_fastmath_raises():
    from xdsl.dialects.builtin import Float32Type

    block = Block(arg_types=[Float32Type()])
    arg = block.args[0]
    op = llvm.FMAOp(arg, arg, arg, llvm.FastMathAttr([FastMathFlag.NO_NANS]))

    with pytest.raises(NotImplementedError, match="Fast-math flags not supported"):
        convert_op(op, MagicMock(), {arg: MagicMock()})


def test_convert_zero():
    op = llvm.ZeroOp.create(result_types=[llvm.LLVMPointerType()])
    val_map: dict[SSAValue, Any] = {}

    convert_op(op, MagicMock(), val_map)

    assert str(val_map[op.res]) == "ptr null"
