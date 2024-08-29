from collections.abc import Sequence

import pytest

from xdsl.dialects.builtin import Float64Type, MemRefType, f64
from xdsl.ir.affine import AffineMap
from xdsl.transforms.convert_memref_stream_to_snitch_stream import (
    strides_for_affine_map,
)
from xdsl.utils.exceptions import DiagnosticException


def mem_type(shape: Sequence[int]) -> MemRefType[Float64Type]:
    return MemRefType(f64, shape)


def test_strides_map_from_memref_type():
    with pytest.raises(
        DiagnosticException,
        match="Unsupported empty shape in memref of type memref<f64>",
    ):
        mem_type([]).get_affine_map_in_bytes()

    assert mem_type([2, 3]).get_affine_map_in_bytes() == AffineMap.from_callable(
        lambda i, j: (i * 24 + j * 8,)
    )


def test_strides_for_affine_map():
    assert strides_for_affine_map(AffineMap.identity(1), mem_type([2])) == [8]
    assert strides_for_affine_map(AffineMap.identity(2), mem_type([2, 3])) == [24, 8]
    assert strides_for_affine_map(AffineMap.identity(2), mem_type([3, 2])) == [16, 8]
    assert strides_for_affine_map(AffineMap.identity(3), mem_type([4, 3, 2])) == [
        48,
        16,
        8,
    ]
    assert strides_for_affine_map(AffineMap.transpose_map(), mem_type([3, 2])) == [
        8,
        16,
    ]
    assert strides_for_affine_map(AffineMap.transpose_map(), mem_type([2, 3])) == [
        8,
        24,
    ]
