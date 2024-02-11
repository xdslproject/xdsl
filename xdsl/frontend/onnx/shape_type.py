from onnx import TensorShapeProto, TypeProto

from xdsl.dialects.builtin import TensorType
from xdsl.frontend.onnx.elem_type import get_elem_type
from xdsl.ir import Attribute


def get_shape(shape: TensorShapeProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in shape.dim)


def get_tensor_type(tensor: TypeProto.Tensor) -> TensorType[Attribute]:
    elem_type = get_elem_type(tensor.elem_type)
    shape = get_shape(tensor.shape)
    return TensorType(elem_type, shape)
