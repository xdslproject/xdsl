import pytest

from xdsl.dialects.builtin import Float32Type, TensorType, f32, f64

pytest.importorskip("onnx", reason="onnx is an optional dependency")

import onnx  # noqa: E402
from onnx import TensorShapeProto, TypeProto  # noqa: E402

from xdsl.frontend.onnx.type import (  # noqa: E402
    get_elem_type,
    get_shape,
    get_tensor_type,
    get_type,
)
from xdsl.utils.hints import isa  # noqa: E402


def test_get_elem_type():
    # test case 1: check if 1 corresponds to f32
    assert get_elem_type(1) == f32

    # test case 11: check if 11 corresponds to f64
    assert get_elem_type(11) == f64

    # test case -1: check if -1 (or other illegal values) corresponds to None
    with pytest.raises(ValueError, match="Unknown elem_type: -1"):
        get_elem_type(-1)


def test_get_type():
    tensor_type = onnx.TypeProto()
    tensor_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    tensor_type.tensor_type.shape.dim.extend(
        [
            onnx.TensorShapeProto.Dimension(dim_value=3),
            onnx.TensorShapeProto.Dimension(dim_value=4),
            onnx.TensorShapeProto.Dimension(dim_value=5),
        ]
    )
    tt = get_type(tensor_type)

    assert isa(tt, TensorType[Float32Type])

    assert tt.get_num_dims() == 3
    assert tt.get_shape() == (3, 4, 5)
    assert tt.get_element_type().name == "f32"


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
