from typing import Any
from unittest.mock import MagicMock

import pytest
from llvmlite import binding, ir

from xdsl.backend.llvm.convert_op import (
    convert_op,
    create_constant,
    declare_intrinsic,
    intrinsic_suffix,
)
from xdsl.dialects import llvm
from xdsl.dialects.builtin import (
    DenseArrayBase,
    FloatAttr,
    IntegerAttr,
    UnitAttr,
    f32,
    f64,
    i32,
    i64,
)
from xdsl.dialects.utils import FastMathFlag
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.utils.exceptions import LLVMTranslationException


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


@pytest.mark.parametrize("intrinsic", [False, True])
@pytest.mark.parametrize("with_operand", [True, False], ids=["operand", "empty-bundle"])
def test_call_op_bundles_raises(intrinsic: bool, with_operand: bool):
    block = Block(arg_types=[i32])
    operands = (block.args[0],) if with_operand else ()
    # A zero-sized bundle still needs support even though it has no operands.
    sizes = DenseArrayBase.from_list(i32, [] if with_operand else [0])
    if intrinsic:
        op = llvm.CallIntrinsicOp(
            "llvm.donothing", [], [], op_bundle_sizes=sizes, op_bundle_operands=operands
        )
    else:
        op = llvm.CallOp("callee", op_bundle_sizes=sizes, op_bundle_operands=operands)
    builder = MagicMock()
    with pytest.raises(NotImplementedError, match="Operand bundles not supported"):
        convert_op(op, builder, {})
    assert not builder.mock_calls


@pytest.mark.parametrize("intrinsic", [False, True])
def test_calls_without_bundles_still_emit(intrinsic: bool):
    module = ir.Module()
    fn_type = ir.FunctionType(ir.VoidType(), [])
    caller = ir.Function(module, fn_type, name="caller")
    block = caller.append_basic_block()
    builder = ir.IRBuilder(block)
    if intrinsic:
        op = llvm.CallIntrinsicOp("llvm.donothing", [], [])
    else:
        ir.Function(module, fn_type, name="callee")
        op = llvm.CallOp("callee")
    convert_op(op, builder, {})
    builder.ret_void()
    assert isinstance(block.instructions[0], ir.CallInstr)
    binding.parse_assembly(str(module)).verify()  # pyright: ignore[reportUnknownMemberType]


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


@pytest.mark.parametrize(
    "ty, expected",
    [
        (ir.FloatType(), "f32"),
        (ir.DoubleType(), "f64"),
        (ir.HalfType(), "f16"),
        (ir.IntType(32), "i32"),
        (ir.VectorType(ir.FloatType(), 4), "v4f32"),
        (ir.VectorType(ir.DoubleType(), 2), "v2f64"),
    ],
)
def test_intrinsic_suffix(ty: ir.Type, expected: str):
    assert intrinsic_suffix(ty) == expected


def test_declare_intrinsic_creates_function():
    module = ir.Module()
    fnty = ir.FunctionType(
        ir.FloatType(), [ir.FloatType(), ir.VectorType(ir.FloatType(), 4)]
    )
    func = declare_intrinsic(
        module, "llvm.vector.reduce.fadd", ir.VectorType(ir.FloatType(), 4), fnty
    )
    assert func.name == "llvm.vector.reduce.fadd.v4f32"


def test_declare_intrinsic_reuses_existing():
    module = ir.Module()
    fnty = ir.FunctionType(
        ir.FloatType(), [ir.FloatType(), ir.VectorType(ir.FloatType(), 4)]
    )
    vec_ty = ir.VectorType(ir.FloatType(), 4)
    first = declare_intrinsic(module, "llvm.vector.reduce.fadd", vec_ty, fnty)
    second = declare_intrinsic(module, "llvm.vector.reduce.fadd", vec_ty, fnty)
    assert first is second


def test_convert_zero():
    op = llvm.ZeroOp.create(result_types=[llvm.LLVMPointerType()])
    val_map: dict[SSAValue, Any] = {}

    convert_op(op, MagicMock(), val_map)

    assert str(val_map[op.res]) == "ptr null"


@pytest.mark.parametrize(
    "xdsl_type, attr, llvmlite_type, value",
    [
        (i32, IntegerAttr(1, i32), ir.IntType(32), 1),
        (
            i64,
            IntegerAttr(123, i64),
            ir.IntType(64),
            123,
        ),
        (f32, FloatAttr(3.0, f32), ir.FloatType(), 3.0),
        (f64, FloatAttr(-42.5, f64), ir.DoubleType(), -42.5),
    ],
)
def test_create_constant(
    xdsl_type: Attribute,
    attr: Attribute,
    llvmlite_type: ir.Type,
    value: object,
) -> None:
    assert create_constant(xdsl_type, attr) == ir.Constant(llvmlite_type, value)


def test_create_constant_invalid():
    with pytest.raises(
        LLVMTranslationException, match="Unsupported constant attribute type"
    ):
        create_constant(i32, UnitAttr())
