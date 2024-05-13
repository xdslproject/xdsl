from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.transforms.convert_memref_stream_to_snitch_stream import (
    offset_map_from_shape,
    strides_for_affine_map,
)


def test_offset_map_from_shape():
    assert offset_map_from_shape([], 5) == AffineMap(0, 0, (AffineExpr.constant(5),))
    assert offset_map_from_shape([2, 3], 5) == AffineMap.from_callable(
        lambda i, j: (i * 15 + j * 5,)
    )


def test_strides_for_affine_map():
    assert strides_for_affine_map(AffineMap.identity(1), [2], 8) == [8]
    assert strides_for_affine_map(AffineMap.identity(2), [2, 3], 8) == [24, 8]
    assert strides_for_affine_map(AffineMap.identity(2), [3, 2], 8) == [16, 8]
    assert strides_for_affine_map(AffineMap.identity(3), [4, 3, 2], 8) == [48, 16, 8]
    assert strides_for_affine_map(AffineMap.transpose_map(), [3, 2], 8) == [8, 16]
    assert strides_for_affine_map(AffineMap.transpose_map(), [2, 3], 8) == [8, 24]
