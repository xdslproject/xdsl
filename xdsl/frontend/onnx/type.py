from onnx import TensorShapeProto, TypeProto

from xdsl.dialects.builtin import TensorType, f32, f64
from xdsl.ir import Attribute

ELEM_TYPE = {
    1: f32,
    11: f64,
}


def get_elem_type(code: int) -> Attribute:
    if code in ELEM_TYPE:
        return ELEM_TYPE[code]
    else:
        raise ValueError(f"Unknown elem_type: {code}")


def get_type(type: TypeProto) -> Attribute:
    tt = get_tensor_type(type.tensor_type)
    return tt


def get_shape(shape: TensorShapeProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in shape.dim)


def get_tensor_type(tensor: TypeProto.Tensor) -> TensorType[Attribute]:
    elem_type = get_elem_type(tensor.elem_type)
    shape = get_shape(tensor.shape)
    return TensorType(elem_type, shape)
