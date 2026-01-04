from unittest.mock import Mock, patch

import pytest

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.dialects import cf, llvm
from xdsl.dialects import vector as xvector
from xdsl.dialects.builtin import (
    DenseArrayBase,
    DenseIntElementsAttr,
    VectorType,
    f32,
    i1,
    i32,
    i64,
)
from xdsl.ir import Attribute, Block, SSAValue


class TestSSAValue(SSAValue):
    def __init__(self, type_attr: Attribute):
        super().__init__(type_attr)

    @property
    def owner(self):
        return None


v_i32 = VectorType(i32, [1])
v_f32 = VectorType(f32, [1])


@pytest.mark.parametrize(
    "op_class, builder_method, arg_types",
    [
        # Arithmetic
        (llvm.AddOp, "add", [i32, i32]),
        (llvm.FAddOp, "fadd", [f32, f32]),
        (llvm.SubOp, "sub", [i32, i32]),
        (llvm.FSubOp, "fsub", [f32, f32]),
        (llvm.MulOp, "mul", [i32, i32]),
        (llvm.FMulOp, "fmul", [f32, f32]),
        (llvm.UDivOp, "udiv", [i32, i32]),
        (llvm.SDivOp, "sdiv", [i32, i32]),
        (llvm.FDivOp, "fdiv", [f32, f32]),
        (llvm.URemOp, "urem", [i32, i32]),
        (llvm.SRemOp, "srem", [i32, i32]),
        (llvm.FRemOp, "frem", [f32, f32]),
        # Bitwise
        (llvm.ShlOp, "shl", [i32, i32]),
        (llvm.LShrOp, "lshr", [i32, i32]),
        (llvm.AShrOp, "ashr", [i32, i32]),
        (llvm.AndOp, "and_", [i32, i32]),
        (llvm.OrOp, "or_", [i32, i32]),
        (llvm.XOrOp, "xor", [i32, i32]),
        # Vector
        (xvector.ExtractElementOp, "extract_element", [v_i32, i32]),
        (xvector.InsertElementOp, "insert_element", [i32, v_i32, i32]),
        (xvector.FMAOp, "fma", [v_f32, v_f32, v_f32]),
    ],
)
def test_convert_standard_op(op_class, builder_method, arg_types):
    builder = Mock()

    # init val_map
    val_map = {}
    args = [TestSSAValue(t) for t in arg_types]
    llvm_args = [Mock() for _ in args]
    for ssa, llvm_val in zip(args, llvm_args):
        val_map[ssa] = llvm_val

    op = op_class(*args)
    convert_op(op, builder, val_map)

    expected_args = list(llvm_args)
    if op_class == xvector.InsertElementOp:
        expected_args[0], expected_args[1] = expected_args[1], expected_args[0]

    getattr(builder, builder_method).assert_called_once_with(*expected_args)
    assert val_map[op.results[0]] == getattr(builder, builder_method).return_value


@pytest.mark.parametrize(
    "op_class, builder_method",
    [
        (llvm.TruncOp, "trunc"),
        (llvm.ZExtOp, "zext"),
        (llvm.SExtOp, "sext"),
        (llvm.PtrToIntOp, "ptrtoint"),
        (llvm.IntToPtrOp, "inttoptr"),
        (xvector.BitcastOp, "bitcast"),
    ],
)
def test_convert_cast_op(op_class, builder_method):
    builder = Mock()
    val_map = {}

    arg = TestSSAValue(i64)
    llvm_arg = Mock()
    val_map[arg] = llvm_arg

    if op_class == llvm.IntToPtrOp:
        op = op_class(arg)
    else:
        op = op_class(arg, i32)

    with patch("xdsl.backend.llvm.convert_op.convert_type") as mock_convert_type:
        convert_op(op, builder, val_map)

        mock_convert_type.assert_called_with(op.results[0].type)
        getattr(builder, builder_method).assert_called_once_with(
            llvm_arg, mock_convert_type.return_value
        )


@pytest.mark.parametrize(
    "op_class, builder_method",
    [
        (llvm.ExtractValueOp, "extract_value"),
        (llvm.InsertValueOp, "insert_value"),
    ],
)
def test_convert_aggregate_op(op_class, builder_method):
    builder = Mock()
    val_map = {}

    container = TestSSAValue(i32)  # Type doesn't matter much for mock
    value = TestSSAValue(i32)
    val_map[container] = Mock()
    val_map[value] = Mock()

    pos = DenseArrayBase.from_list(i64, [1, 2])

    call_args = []
    if op_class == llvm.ExtractValueOp:
        op = op_class(pos, container, i32)
        call_args = [val_map[container], [1, 2]]
    else:
        op = op_class(pos, container, value)
        call_args = [val_map[container], val_map[value], [1, 2]]

    convert_op(op, builder, val_map)
    getattr(builder, builder_method).assert_called_once_with(*call_args)
    assert val_map[op.results[0]] == getattr(builder, builder_method).return_value


@pytest.mark.parametrize("is_load", [True, False])
def test_memory_access(is_load):
    builder = Mock()
    val_map = {}
    ptr = TestSSAValue(i32)
    val_map[ptr] = Mock()

    if is_load:
        op = llvm.LoadOp(ptr, i32)
        convert_op(op, builder, val_map)
        builder.load.assert_called_once_with(val_map[ptr])
        assert val_map[op.results[0]] == builder.load.return_value
    else:
        val = TestSSAValue(i32)
        val_map[val] = Mock()
        op = llvm.StoreOp(val, ptr)
        convert_op(op, builder, val_map)
        builder.store.assert_called_once_with(val_map[val], val_map[ptr])


def test_alloca():
    builder = Mock()
    val_map = {}
    with patch("xdsl.backend.llvm.convert_op.convert_type") as mock_convert_type:
        size = TestSSAValue(i64)
        llvm_size = Mock()
        val_map[size] = llvm_size
        op = llvm.AllocaOp(size, i32)
        convert_op(op, builder, val_map)
        builder.alloca.assert_called_once_with(
            mock_convert_type.return_value, size=llvm_size
        )


def test_gep():
    builder = Mock()
    val_map = {}
    ptr = TestSSAValue(i32)
    idx_ssa = TestSSAValue(i32)
    llvm_ptr = Mock()
    llvm_idx = Mock()
    val_map[ptr] = llvm_ptr
    val_map[idx_ssa] = llvm_idx

    op = llvm.GEPOp(
        ptr, [llvm.GEP_USE_SSA_VAL, 42], i32, ssa_indices=[idx_ssa], inbounds=True
    )

    with patch("xdsl.backend.llvm.convert_op.convert_type"):
        convert_op(op, builder, val_map)

    args = builder.gep.call_args[0]
    assert args[1][0] == llvm_idx
    assert args[1][1].constant == 42
    assert builder.gep.call_args[1]["inbounds"] is True


def test_vector_shuffle():
    builder = Mock()
    val_map = {}
    v_type = VectorType(i32, [4])
    v1 = TestSSAValue(v_type)
    v2 = TestSSAValue(v_type)
    val_map[v1] = Mock()
    val_map[v2] = Mock()

    op = xvector.ShuffleOp(
        v1, v2, DenseArrayBase.from_list(i64, [0, 1, 0, 1]), result_type=v_type
    )
    convert_op(op, builder, val_map)

    args = builder.shuffle_vector.call_args[0]
    assert [c.constant for c in args[2].constant] == [0, 1, 0, 1]


def test_control_flow():
    builder = Mock()
    val_map = {}
    block_map = {}

    # Branch
    block = Block()
    llvm_block = Mock()
    block_map[block] = llvm_block

    op = cf.BranchOp(block)
    convert_op(op, builder, val_map, block_map)
    builder.branch.assert_called_once_with(llvm_block)

    # Conditional Branch
    builder.reset_mock()
    cond = TestSSAValue(i1)
    b1 = Block()
    b2 = Block()
    llvm_cond = Mock()
    llvm_b1 = Mock()
    llvm_b2 = Mock()

    val_map[cond] = llvm_cond
    block_map[b1] = llvm_b1
    block_map[b2] = llvm_b2

    op = cf.ConditionalBranchOp(cond, b1, [], b2, [])
    convert_op(op, builder, val_map, block_map)
    builder.cbranch.assert_called_once_with(llvm_cond, llvm_b1, llvm_b2)


def test_switch():
    builder = Mock()
    val_map = {}
    block_map = {}
    flag = TestSSAValue(i32)
    default_block = Block()
    case_block = Block()
    llvm_flag = Mock()
    llvm_default = Mock()
    llvm_case = Mock()
    val_map[flag] = llvm_flag
    block_map[default_block] = llvm_default
    block_map[case_block] = llvm_case

    op = cf.SwitchOp(
        flag,
        default_block,
        [],
        case_values=DenseIntElementsAttr.from_list(VectorType(i32, [1]), [0]),
        case_blocks=[case_block],
    )
    switch_inst = Mock()
    builder.switch.return_value = switch_inst

    convert_op(op, builder, val_map, block_map)
    builder.switch.assert_called_once_with(llvm_flag, llvm_default)
    switch_inst.add_case.assert_called_once_with(0, llvm_case)


def test_func_call():
    builder = Mock()
    val_map = {}
    module = Mock()
    builder.module = module
    callee = Mock()
    module.get_global.return_value = callee

    arg = TestSSAValue(i32)
    llvm_arg = Mock()
    val_map[arg] = llvm_arg

    op = llvm.CallOp("my_func", arg, return_type=i32)
    convert_op(op, builder, val_map)

    builder.call.assert_called_once_with(callee, [llvm_arg])


def test_inline_asm():
    builder = Mock()
    val_map = {}
    arg = TestSSAValue(i32)
    llvm_arg = Mock()
    val_map[arg] = llvm_arg

    op = llvm.InlineAsmOp("nop", "=r,r", [arg], [i32])

    with patch("xdsl.backend.llvm.convert_op.convert_type") as m_ct:
        convert_op(op, builder, val_map)
        m_ct.assert_any_call(i32)
        builder.call.assert_called()


def test_unreachable():
    builder = Mock()
    op = llvm.UnreachableOp()
    convert_op(op, builder, {})
    builder.unreachable.assert_called_once()


def test_convert_return():
    builder = Mock()
    val_map = {}

    # Void return
    op = llvm.ReturnOp()
    convert_op(op, builder, val_map)
    builder.ret_void.assert_called_once()

    # Value return
    builder.reset_mock()
    val = TestSSAValue(i32)
    llvm_val = Mock()
    val_map[val] = llvm_val
    op = llvm.ReturnOp(val)
    convert_op(op, builder, val_map)
    builder.ret.assert_called_once_with(llvm_val)
