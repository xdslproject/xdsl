import pytest

from xdsl.dialects.builtin import TensorType, f32, f64

try:
    from onnx import TensorShapeProto, TypeProto  # noqa: E402

    from xdsl.frontend.onnx.shape_type import get_shape, get_tensor_type  # noqa: E402
except ImportError as exc:
    print(exc)
    pytest.skip("onnx is an optional dependency", allow_module_level=True)


def test_get_shape():
    assert get_shape(TensorShapeProto()) == ()
    assert get_shape(
        TensorShapeProto(dim=(TensorShapeProto.Dimension(dim_value=1),))
    ) == (1,)
    assert get_shape(
        TensorShapeProto(
            dim=(
                TensorShapeProto.Dimension(dim_value=1),
                TensorShapeProto.Dimension(dim_value=2),
            )
        )
    ) == (1, 2)


def test_get_tensor_type():
    assert get_tensor_type(
        TypeProto.Tensor(
            elem_type=1,
            shape=TensorShapeProto(
                dim=(
                    TensorShapeProto.Dimension(dim_value=2),
                    TensorShapeProto.Dimension(dim_value=3),
                )
            ),
        )
    ) == TensorType(f32, (2, 3))
    assert get_tensor_type(
        TypeProto.Tensor(
            elem_type=11,
            shape=TensorShapeProto(
                dim=(
                    TensorShapeProto.Dimension(dim_value=4),
                    TensorShapeProto.Dimension(dim_value=5),
                )
            ),
        )
    ) == TensorType(f64, (4, 5))
