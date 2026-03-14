from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseArrayBase,
    IndexType,
    TensorType,
    UnitAttr,
    f32,
    f64,
    i64,
)
from xdsl.dialects.stencil import IndexAttr
from xdsl.dialects.tensor import ExtractSliceOp, InsertSliceOp, PadOp, YieldOp
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value


def test_extract_slice_static():
    input_t = TensorType(f64, [10, 20, 30])
    input_v = TestOp(result_types=[input_t]).res[0]

    extract_slice = ExtractSliceOp.from_static_parameters(input_v, [1, 2, 3], [4, 5, 6])

    assert extract_slice.source is input_v
    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
    assert extract_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5, 6])
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1, 1])
    assert extract_slice.offsets == ()
    assert extract_slice.sizes == ()
    assert extract_slice.strides == ()
    assert extract_slice.result.type == TensorType(f64, [4, 5, 6])

    extract_slice = ExtractSliceOp.from_static_parameters(
        input_v, [1, 2, 3], [4, 5, 6], [8, 9, 10]
    )

    assert extract_slice.source is input_v
    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
    assert extract_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5, 6])
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [8, 9, 10])
    assert extract_slice.offsets == ()
    assert extract_slice.sizes == ()
    assert extract_slice.strides == ()
    assert extract_slice.result.type == TensorType(f64, [4, 5, 6])


def test_insert_slice_static():
    source_t = TensorType(f64, [10, 20])
    source_v = TestOp(result_types=[source_t]).res[0]
    dest_t = TensorType(f64, [10, 20, 30])
    dest_v = TestOp(result_types=[dest_t]).res[0]

    insert_slice = InsertSliceOp.from_static_parameters(
        source_v, dest_v, [1, 2], [4, 5]
    )

    assert insert_slice.source is source_v
    assert insert_slice.dest is dest_v
    assert insert_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2])
    assert insert_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5])
    assert insert_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1])
    assert insert_slice.offsets == ()
    assert insert_slice.sizes == ()
    assert insert_slice.strides == ()
    assert insert_slice.result.type == dest_t

    insert_slice = InsertSliceOp.from_static_parameters(
        source_v, dest_v, [1, 2], [4, 5], [8, 9]
    )

    assert insert_slice.source is source_v
    assert insert_slice.dest is dest_v
    assert insert_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2])
    assert insert_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5])
    assert insert_slice.static_strides == DenseArrayBase.from_list(i64, [8, 9])
    assert insert_slice.offsets == ()
    assert insert_slice.sizes == ()
    assert insert_slice.strides == ()
    assert insert_slice.result.type == dest_t


def test_insert_slice_dynamic():
    source_t = TensorType(f64, [10, 20])
    source_v = create_ssa_value(source_t)
    dest_t = TensorType(f64, [10, 20, 30])
    dest_v = create_ssa_value(dest_t)
    offset1 = create_ssa_value(IndexAttr.get(3))
    offset2 = create_ssa_value(IndexAttr.get(15))
    stride1 = create_ssa_value(IndexAttr.get(2))
    stride2 = create_ssa_value(IndexAttr.get(5))

    insert_slice = InsertSliceOp.get(
        source=source_v,
        dest=dest_v,
        static_sizes=[1, 2],
        offsets=[offset1, offset2],
        strides=[stride1, stride2],
    )

    assert insert_slice.static_offsets == DenseArrayBase.from_list(
        i64, 2 * [DYNAMIC_INDEX]
    )
    assert insert_slice.static_strides == DenseArrayBase.from_list(
        i64, 2 * [DYNAMIC_INDEX]
    )


def test_pad_op_static():
    source_t = TensorType(f32, [2, 3])
    source_v = TestOp(result_types=[source_t]).res[0]
    pval_v = TestOp(result_types=[f32]).res[0]

    yield_op = YieldOp(pval_v)
    block = Block(arg_types=[IndexType(), IndexType()])
    block.add_op(yield_op)
    region = Region([block])

    result_t = TensorType(f32, [6, 9])
    pad_op = PadOp(source_v, [], [], region, [1, 2], [3, 4], None, result_t)

    assert pad_op.source is source_v
    assert pad_op.static_low == DenseArrayBase.from_list(i64, [1, 2])
    assert pad_op.static_high == DenseArrayBase.from_list(i64, [3, 4])
    assert pad_op.low == ()
    assert pad_op.high == ()
    assert pad_op.nofold is None
    assert pad_op.result.type == result_t


def test_pad_op_dynamic():
    source_t = TensorType(f32, [2, 3])
    source_v = TestOp(result_types=[source_t]).res[0]
    pval_v = TestOp(result_types=[f32]).res[0]
    low_dyn = create_ssa_value(IndexType())
    high_dyn = create_ssa_value(IndexType())

    yield_op = YieldOp(pval_v)
    block = Block(arg_types=[IndexType(), IndexType()])
    block.add_op(yield_op)
    region = Region([block])

    result_t = TensorType(f32, [DYNAMIC_INDEX, 9])
    pad_op = PadOp(
        source_v,
        [low_dyn],
        [high_dyn],
        region,
        [DYNAMIC_INDEX, 2],
        [3, DYNAMIC_INDEX],
        UnitAttr(),
        result_t,
    )

    assert pad_op.source is source_v
    assert pad_op.static_low == DenseArrayBase.from_list(i64, [DYNAMIC_INDEX, 2])
    assert pad_op.static_high == DenseArrayBase.from_list(i64, [3, DYNAMIC_INDEX])
    assert len(pad_op.low) == 1
    assert len(pad_op.high) == 1
    assert pad_op.low[0] is low_dyn
    assert pad_op.high[0] is high_dyn
    assert pad_op.nofold == UnitAttr()
    assert pad_op.result.type == result_t


def test_yield_op():
    pval_v = TestOp(result_types=[f32]).res[0]

    yield_op = YieldOp(pval_v)

    assert yield_op.name == "tensor.yield"
    assert len(yield_op.arguments) == 1
    assert yield_op.arguments[0] is pval_v
