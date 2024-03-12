import onnx
import pytest

from xdsl.frontend.onnx.type import f32, f64, get_elem_type, get_shape, get_tensor_type


def test_get_elem_type():
    # test case 1: check if 1 corresponds to f32
    assert get_elem_type(1) == f32

    # test case 11: check if 11 corresponds to f64
    assert get_elem_type(11) == f64

    # test case -1: check if -1 (or other illegal values) corresponds to None
    with pytest.raises(ValueError, match="Unknown elem_type: -1"):
        get_elem_type(-1)


def test_get_type():
    pass


def test_get_shape():
    tensor_shape = onnx.TensorShapeProto()
    tensor_shape.dim.extend(
        [
            onnx.TensorShapeProto.Dimension(dim_value=3),
            onnx.TensorShapeProto.Dimension(dim_value=4),
            onnx.TensorShapeProto.Dimension(dim_value=5),
        ]
    )
    shape = get_shape(tensor_shape)
    assert len(shape) == 3
    assert shape[0] == 3
    assert shape[1] == 4
    assert shape[2] == 5


def test_get_tensor_type():
    tensor_type = onnx.TypeProto()
    tensor_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    tensor_type.tensor_type.shape.dim.extend(
        [
            onnx.TensorShapeProto.Dimension(dim_value=3),
            onnx.TensorShapeProto.Dimension(dim_value=4),
            onnx.TensorShapeProto.Dimension(dim_value=5),
        ]
    )

    tt = get_tensor_type(tensor_type.tensor_type)
    assert tt.get_element_type().name == "f32"
    shape = tt.get_shape()
    n_dim = len(shape)
    assert n_dim == 3
    assert shape[0] == 3
    assert shape[1] == 4
    assert shape[2] == 5
