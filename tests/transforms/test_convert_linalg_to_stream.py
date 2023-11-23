from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.transforms.convert_linalg_to_stream import (
    offset_map_from_shape,
    strides_for_affine_map,
)


def test_offset_map_from_shape():
    assert offset_map_from_shape([]) == AffineMap(0, 0, (AffineExpr.constant(1),))
    assert offset_map_from_shape([2, 3]) == AffineMap.from_callable(
        lambda i, j: (i * 3 + j,)
    )


def test_strides_for_affine_map():
    assert strides_for_affine_map(AffineMap.identity(1), [2]) == [1]
    assert strides_for_affine_map(AffineMap.identity(2), [2, 3]) == [3, 1]
    assert strides_for_affine_map(AffineMap.identity(2), [3, 2]) == [2, 1]
    assert strides_for_affine_map(AffineMap.identity(3), [4, 3, 2]) == [6, 2, 1]
    assert strides_for_affine_map(AffineMap.transpose_map(), [3, 2]) == [1, 2]
    assert strides_for_affine_map(AffineMap.transpose_map(), [2, 3]) == [1, 3]
