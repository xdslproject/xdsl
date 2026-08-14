import pytest

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseArrayBase,
    IndexType,
    TensorType,
    f64,
    i64,
)
from xdsl.dialects.stencil import IndexAttr
from xdsl.dialects.tensor import ExtractSliceOp, FromElementsOp, InsertSliceOp
from xdsl.dialects.test import TestOp
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


def test_extract_slice_infer_result_type():
    source_t = TensorType(f64, [10, 20, 30])
    size = create_ssa_value(IndexType())

    assert ExtractSliceOp.infer_result_type(source_t, [4, 5, 6]) == TensorType(
        f64, [4, 5, 6]
    )

    # A dynamic size makes that dimension dynamic in the result type.
    assert ExtractSliceOp.infer_result_type(source_t, [size, 5, 6]) == TensorType(
        f64, [DYNAMIC_INDEX, 5, 6]
    )


def test_extract_slice_infer_result_type_rank_mismatch():
    source_t = TensorType(f64, [10, 20, 30])

    with pytest.raises(ValueError, match="sizes to match source rank"):
        ExtractSliceOp.infer_result_type(source_t, [4, 5])


def test_extract_slice_dynamic():
    source_t = TensorType(f64, [10, 20, 30])
    source_v = create_ssa_value(source_t)
    offset = create_ssa_value(IndexType())
    size = create_ssa_value(IndexType())

    result_type = ExtractSliceOp.infer_result_type(source_t, [size, 5, 6])
    extract_slice = ExtractSliceOp(
        source_v, [offset, 0, 0], [size, 5, 6], [1, 1, 1], result_type
    )
    extract_slice.verify()

    # Static entries keep their value; dynamic ones are marked and moved to operands.
    assert extract_slice.static_offsets == DenseArrayBase.from_list(
        i64, [DYNAMIC_INDEX, 0, 0]
    )
    assert extract_slice.static_sizes == DenseArrayBase.from_list(
        i64, [DYNAMIC_INDEX, 5, 6]
    )
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1, 1])

    assert extract_slice.offsets == (offset,)
    assert extract_slice.sizes == (size,)
    assert extract_slice.strides == ()

    assert extract_slice.result.type == TensorType(f64, [DYNAMIC_INDEX, 5, 6])


def test_extract_slice_get_all_static():
    source_t = TensorType(f64, [10, 20, 30])
    source_v = create_ssa_value(source_t)

    result_type = ExtractSliceOp.infer_result_type(source_t, [4, 5, 6])
    extract_slice = ExtractSliceOp(
        source_v, [1, 2, 3], [4, 5, 6], [1, 1, 1], result_type
    )
    extract_slice.verify()

    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
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
    offset1 = create_ssa_value(IndexAttr.from_indices(3))
    offset2 = create_ssa_value(IndexAttr.from_indices(15))
    stride1 = create_ssa_value(IndexAttr.from_indices(2))
    stride2 = create_ssa_value(IndexAttr.from_indices(5))

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


def test_insert_element_init():
    test_op = TestOp(result_types=(i64, i64, f64, f64))
    i64_0, i64_1, f64_0, f64_1 = test_op.results

    # Scalar
    assert FromElementsOp(
        i64_0, result_type=TensorType(i64, ())
    ).result.type == TensorType(i64, ())
    # 1xi64
    assert FromElementsOp(i64_0).result.type == TensorType(i64, (1,))
    # 2xi64
    assert FromElementsOp(i64_0, i64_1).result.type == TensorType(i64, (2,))
    # 2xi64 Splat
    assert FromElementsOp(*(i64_0, i64_1)).result.type == TensorType(i64, (2,))

    # 2xf64
    assert FromElementsOp(f64_0, f64_1).result.type == TensorType(f64, (2,))
