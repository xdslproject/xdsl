import re
from collections.abc import Sequence

import pytest

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseArrayBase,
    IntegerAttr,
    TensorType,
    f32,
    f64,
    i64,
)
from xdsl.dialects.stencil import IndexAttr
from xdsl.dialects.tensor import ConcatOp, ExtractSliceOp, InsertSliceOp
from xdsl.dialects.test import TestOp
from xdsl.utils.exceptions import VerifyException
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


@pytest.mark.parametrize(
    ("arg_types", "result_type", "dim", "exp_error"),
    [
        ([TensorType(f32, (1, 2, 3))], TensorType(f32, (1, 2, 3)), 0, None),
        ([TensorType(f32, (1, 2, 3))], TensorType(f32, (1, 2, 3)), 2, None),
        (
            [TensorType(f32, (1, 2, 3))],
            TensorType(f32, (1, 2, 3)),
            3,
            "concatenation dim must be less than the tensor rank",
        ),
        (
            [TensorType(f32, (1, 2, 3)), TensorType(f32, (1, 2, 3))],
            TensorType(f32, (2, 2, 3)),
            0,
            None,
        ),
        (
            [TensorType(f32, (1, 2, 3)), TensorType(f32, (1, 2, 3))],
            TensorType(f32, (1, 1, 3)),
            0,
            re.escape(
                "result type tensor<1x1x3xf32> does not match inferred shape [2, 2, 3] static sizes"
            ),
        ),
        (
            [TensorType(f32, (1,)), TensorType(f32, (9999,))],
            TensorType(f32, (10000,)),
            0,
            None,
        ),
        (
            [TensorType(f32, (1,)), TensorType(f32, (9999,))],
            TensorType(f32, (DYNAMIC_INDEX,)),
            0,
            None,
        ),
        (
            [TensorType(f32, (1,)), TensorType(f32, (9999,))],
            TensorType(f32, (10001,)),
            0,
            re.escape(
                "result type tensor<10001xf32> does not match inferred shape [10000] static sizes"
            ),
        ),
        (
            [TensorType(f32, (1, 2)), TensorType(f32, (1, 2)), TensorType(f32, (2, 1))],
            TensorType(f32, (4, 2)),
            0,
            "static concatenation size mismatch along non-concatenated dimension 1",
        ),
        (
            [
                TensorType(f32, (1, 2)),
                TensorType(f32, (1, 2)),
                TensorType(f32, (2, DYNAMIC_INDEX)),
            ],
            TensorType(f32, (4, 2)),
            0,
            None,
        ),
        (
            [
                TensorType(f32, (1, 2)),
                TensorType(f32, (1, 2)),
                TensorType(f32, (2, DYNAMIC_INDEX)),
            ],
            TensorType(f32, (4, DYNAMIC_INDEX)),
            0,
            None,
        ),
        (
            [
                TensorType(f32, (2, 2)),
                TensorType(f32, (2, 2)),
                TensorType(f32, (2, DYNAMIC_INDEX)),
            ],
            TensorType(f32, (2, DYNAMIC_INDEX)),
            1,
            None,
        ),
        (
            [
                TensorType(f32, (3, 2)),
                TensorType(f32, (3, 2)),
                TensorType(f32, (3, DYNAMIC_INDEX)),
            ],
            TensorType(f32, (3, 1000)),
            1,
            None,
        ),
    ],
)
def test_concat_verifies(
    arg_types: Sequence[TensorType],
    result_type: TensorType,
    dim: int | IntegerAttr,
    exp_error: str | None,
) -> None:
    """Test that ConcatOp's custom verify method correctly raises exceptions"""
    operands = TestOp(result_types=arg_types).res
    op = ConcatOp(operands, dim, result_type)
    if exp_error is not None:
        with pytest.raises(VerifyException, match=exp_error):
            op.verify()
    else:
        op.verify()
