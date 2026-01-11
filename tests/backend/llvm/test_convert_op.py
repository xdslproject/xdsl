from collections.abc import Mapping
from typing import Protocol

import llvmlite.ir as ir
import pytest

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseArrayBase, Float32Type, Float64Type, IntegerAttr
from xdsl.ir import Attribute, Operation, SSAValue

#
# common
#


def _convert_and_verify(
    op: Operation,
    expected_ir: str,
    val_map_setup: Mapping[SSAValue, ir.Value] | None = None,
) -> None:
    module = ir.Module(name="test_module")
    func_type = ir.FunctionType(ir.VoidType(), [])
    func = ir.Function(module, func_type, name="test_func")
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    val_map = dict(val_map_setup) if val_map_setup else {}

    convert_op(op, builder, val_map)

    assert len(block.instructions) == 1
    inst_str = str(block.instructions[-1]).strip()
    if "=" in inst_str and "=" not in expected_ir:
        assert inst_str.split(" = ", 1)[1].strip() == expected_ir.strip()
    else:
        assert inst_str == expected_ir.strip()


def _create_ssa_value(typ: Attribute) -> SSAValue:
    class TestOp(Operation):
        name = "test.op"

    return TestOp(result_types=[typ]).results[0]


i32 = llvm.i32
i64 = llvm.i64
f32 = Float32Type()
ptr_ty = llvm.LLVMPointerType()

val_i32 = _create_ssa_value(i32)
val_i64 = _create_ssa_value(i64)
val_f32 = _create_ssa_value(f32)
val_ptr = _create_ssa_value(ptr_ty)

mock_i32 = ir.Constant(ir.IntType(32), 1)
mock_i64 = ir.Constant(ir.IntType(64), 2)
mock_f32 = ir.Constant(ir.FloatType(), 3.0)
mock_ptr = ir.Constant(ir.IntType(32).as_pointer(), 0)

default_val_map: dict[SSAValue, ir.Value] = {
    val_i32: mock_i32,
    val_i64: mock_i64,
    val_f32: mock_f32,
    val_ptr: mock_ptr,
}

val_a = val_i32
val_b = _create_ssa_value(i32)
default_val_map[val_b] = ir.Constant(ir.IntType(32), 2)


#
# tests
#


class BinaryOpCtor(Protocol):
    def __call__(self, lhs: SSAValue, rhs: SSAValue, /) -> Operation: ...


@pytest.mark.parametrize(
    "op_type, expected_inst",
    [
        (llvm.AddOp, "add i32 1, 2"),
        (llvm.FAddOp, "fadd i32 1, 2"),
        (llvm.SubOp, "sub i32 1, 2"),
        (llvm.FSubOp, "fsub i32 1, 2"),
        (llvm.MulOp, "mul i32 1, 2"),
        (llvm.FMulOp, "fmul i32 1, 2"),
        (llvm.UDivOp, "udiv i32 1, 2"),
        (llvm.SDivOp, "sdiv i32 1, 2"),
        (llvm.FDivOp, "fdiv i32 1, 2"),
        (llvm.URemOp, "urem i32 1, 2"),
        (llvm.SRemOp, "srem i32 1, 2"),
        (llvm.FRemOp, "frem i32 1, 2"),
        (llvm.ShlOp, "shl i32 1, 2"),
        (llvm.LShrOp, "lshr i32 1, 2"),
        (llvm.AShrOp, "ashr i32 1, 2"),
        (llvm.AndOp, "and i32 1, 2"),
        (llvm.OrOp, "or i32 1, 2"),
        (llvm.XOrOp, "xor i32 1, 2"),
    ],
)
def test_binary_ops(op_type: BinaryOpCtor, expected_inst: str):
    op = op_type(val_a, val_b)
    _convert_and_verify(op, expected_inst, default_val_map)


def test_trunc_op():
    op = llvm.TruncOp(val_i64, i32)
    _convert_and_verify(op, "trunc i64 2 to i32", default_val_map)


def test_zext_op():
    op = llvm.ZExtOp(val_i32, i64)
    _convert_and_verify(op, "zext i32 1 to i64", default_val_map)


def test_sext_op():
    op = llvm.SExtOp(val_i32, i64)
    _convert_and_verify(op, "sext i32 1 to i64", default_val_map)


@pytest.fixture
def prog() -> tuple[ir.IRBuilder, ir.Block]:
    module = ir.Module()
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), []), "main")
    block = func.append_basic_block("entry")
    builder = ir.IRBuilder(block)
    return builder, block


def test_load_op(prog: tuple[ir.IRBuilder, ir.Block]):
    builder, block = prog
    op_alloca = llvm.AllocaOp(val_i32, elem_type=i32)
    ptr_val = op_alloca.results[0]
    op = llvm.LoadOp(ptr_val, i32)

    count_before = len(block.instructions)
    val_map: dict[SSAValue, ir.Value] = {val_i32: mock_i32}
    convert_op(op_alloca, builder, val_map)
    convert_op(op, builder, val_map)

    assert len(block.instructions) == count_before + 2

    inst_str = str(block.instructions[-1]).strip()
    assert " = load i32, i32* %" in inst_str


def test_store_op(prog: tuple[ir.IRBuilder, ir.Block]):
    builder, block = prog
    op_alloca = llvm.AllocaOp(val_i32, elem_type=i32)
    ptr_val = op_alloca.results[0]
    op = llvm.StoreOp(val_i32, ptr_val)

    count_before = len(block.instructions)
    val_map: dict[SSAValue, ir.Value] = {val_i32: mock_i32}
    convert_op(op_alloca, builder, val_map)
    convert_op(op, builder, val_map)

    assert len(block.instructions) == count_before + 2

    inst_str = str(block.instructions[-1]).strip()
    assert inst_str.startswith("store i32 1, i32* %")


@pytest.mark.parametrize("inbounds", [False, True])
def test_gep_op(prog: tuple[ir.IRBuilder, ir.Block], inbounds: bool):
    builder, block = prog
    op_alloca = llvm.AllocaOp(val_i32, elem_type=i32)
    ptr_val = op_alloca.results[0]

    op = llvm.GEPOp(
        ptr_val,
        indices=[llvm.GEP_USE_SSA_VAL],
        pointee_type=i32,
        ssa_indices=[val_a],
        result_type=ptr_ty,
        inbounds=inbounds,
    )

    val_map: dict[SSAValue, ir.Value] = {val_i32: mock_i32, val_a: mock_i32}
    convert_op(op_alloca, builder, val_map)
    convert_op(op, builder, val_map)

    inbounds_str = "inbounds " if inbounds else ""
    inst_str = str(block.instructions[-1]).strip()
    expected_part = f" = getelementptr {inbounds_str}i32, i32* %"
    assert expected_part in inst_str
    assert inst_str.endswith(", i32 1")


def test_gep_constant_idx(prog: tuple[ir.IRBuilder, ir.Block]):
    builder, block = prog
    op_alloca = llvm.AllocaOp(val_i32, elem_type=i32)
    ptr_val = op_alloca.results[0]

    op = llvm.GEPOp(
        ptr_val,
        indices=[0],
        pointee_type=i32,
        result_type=ptr_ty,
    )

    val_map: dict[SSAValue, ir.Value] = {val_i32: mock_i32}
    convert_op(op_alloca, builder, val_map)
    convert_op(op, builder, val_map)

    inst_str = str(block.instructions[-1]).strip()
    assert inst_str.endswith(", i32 0")


def test_ptrtoint_op(prog: tuple[ir.IRBuilder, ir.Block]):
    builder, block = prog
    op_alloca = llvm.AllocaOp(val_i32, elem_type=i32)
    ptr_val = op_alloca.results[0]
    op = llvm.PtrToIntOp(ptr_val, i64)

    count_before = len(block.instructions)
    val_map: dict[SSAValue, ir.Value] = {val_i32: mock_i32}
    convert_op(op_alloca, builder, val_map)
    convert_op(op, builder, val_map)

    assert len(block.instructions) == count_before + 2

    inst_str = str(block.instructions[-1]).strip()
    assert " = ptrtoint i32* %" in inst_str
    assert inst_str.endswith(" to i64")


def test_fpext_op(prog: tuple[ir.IRBuilder, ir.Block]):
    builder, block = prog
    op = llvm.FPExtOp(val_f32, Float64Type())

    count_before = len(block.instructions)
    convert_op(op, builder, default_val_map.copy())

    assert len(block.instructions) == count_before + 1
    inst_str = str(block.instructions[-1]).strip()

    assert "fpext float" in inst_str
    assert " to double" in inst_str


def test_extract_value_op():
    struct_ty = llvm.LLVMStructType.from_type_list([i32, i32])
    agg_val = _create_ssa_value(struct_ty)
    mock_agg = ir.Constant(
        ir.LiteralStructType([ir.IntType(32), ir.IntType(32)]),
        [ir.IntType(32)(1), ir.IntType(32)(2)],
    )

    op = llvm.ExtractValueOp(DenseArrayBase.from_list(i64, [0]), agg_val, i32)
    expected = "extractvalue {i32, i32} {i32 1, i32 2}, 0"
    _convert_and_verify(op, expected, {agg_val: mock_agg})


def test_insert_value_op():
    struct_ty = llvm.LLVMStructType.from_type_list([i32, i32])
    agg_val = _create_ssa_value(struct_ty)
    mock_agg = ir.Constant(
        ir.LiteralStructType([ir.IntType(32), ir.IntType(32)]),
        [ir.IntType(32)(1), ir.IntType(32)(2)],
    )

    op = llvm.InsertValueOp(DenseArrayBase.from_list(i64, [1]), agg_val, val_a)
    expected = "insertvalue {i32, i32} {i32 1, i32 2}, i32 1, 1"
    _convert_and_verify(op, expected, {agg_val: mock_agg, val_a: mock_i32})


@pytest.mark.parametrize("has_ret_val", [False, True])
def test_return_op(has_ret_val: bool):
    if has_ret_val:
        op = llvm.ReturnOp(val_a)
        expected = "ret i32 1"
        _convert_and_verify(op, expected, default_val_map)
    else:
        op = llvm.ReturnOp()
        expected = "ret void"
        _convert_and_verify(op, expected)


def test_call_op_void():
    op = llvm.CallOp("my_func", return_type=None)

    module = ir.Module()
    func_ty = ir.FunctionType(ir.VoidType(), [])
    callee = ir.Function(module, func_ty, "my_func")

    main_func = ir.Function(module, func_ty, "main")
    block = main_func.append_basic_block("entry")
    builder = ir.IRBuilder(block)

    val_map: dict[SSAValue, ir.Value] = {}
    convert_op(op, builder, val_map)
    assert block.instructions[-1].operands[0] == callee
    assert str(block.instructions[-1]).strip() == 'call void @"my_func"()'


def test_call_op_val():
    op = llvm.CallOp("my_func", val_a, return_type=i32)

    module = ir.Module()
    func_ty = ir.FunctionType(ir.IntType(32), [ir.IntType(32)])
    callee = ir.Function(module, func_ty, "my_func")

    main_func = ir.Function(module, ir.FunctionType(ir.VoidType(), []), "main")
    block = main_func.append_basic_block("entry")
    builder = ir.IRBuilder(block)

    val_map: dict[SSAValue, ir.Value] = {val_a: mock_i32}
    convert_op(op, builder, val_map)

    assert block.instructions[-1].operands[0] == callee
    inst_str = str(block.instructions[-1]).strip()
    assert inst_str.split(" = ", 1)[1].strip() == 'call i32 @"my_func"(i32 1)'


def test_inline_asm_op():
    op = llvm.InlineAsmOp("nop", "", [val_a], [i32], has_side_effects=True)
    expected = 'call i32 asm sideeffect "nop", ""(i32 1)'
    _convert_and_verify(op, expected, default_val_map)


def test_inline_asm_void():
    op = llvm.InlineAsmOp("nop", "", [val_a], [], has_side_effects=True)
    expected = 'call void asm sideeffect "nop", ""(i32 1)'
    _convert_and_verify(op, expected, default_val_map)


@pytest.mark.parametrize(
    "predicate, expected_cond",
    [
        ("eq", "eq"),
        ("ne", "ne"),
        ("slt", "slt"),
        ("sle", "sle"),
        ("sgt", "sgt"),
        ("sge", "sge"),
        ("ult", "ult"),
        ("ule", "ule"),
        ("ugt", "ugt"),
        ("uge", "uge"),
    ],
)
def test_icmp_op(predicate: str, expected_cond: str):
    pred_map = {
        "eq": 0,
        "ne": 1,
        "slt": 2,
        "sle": 3,
        "sgt": 4,
        "sge": 5,
        "ult": 6,
        "ule": 7,
        "ugt": 8,
        "uge": 9,
    }

    op = llvm.ICmpOp(
        val_a, val_b, IntegerAttr.from_int_and_width(pred_map[predicate], 64)
    )
    expected = f"icmp {expected_cond} i32 1, 2"
    _convert_and_verify(op, expected, default_val_map)


def test_unreachable_op():
    op = llvm.UnreachableOp()
    expected = "unreachable"
    _convert_and_verify(op, expected)


def test_unsupported_op(prog: tuple[ir.IRBuilder, ir.Block]):
    builder, _ = prog

    class UnknownOp(Operation):
        name = "llvm.unknown"

    op = UnknownOp()

    with pytest.raises(
        NotImplementedError, match="Conversion not implemented for op: llvm.unknown"
    ):
        convert_op(op, builder, {})
