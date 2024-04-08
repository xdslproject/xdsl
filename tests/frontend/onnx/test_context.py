import onnx
import pytest

from xdsl.ir import Attribute

try:
    from onnx import TensorProto, ValueInfoProto

    from xdsl.frontend.onnx.context import (
        OnnxXdslMapping,
        visit_value_info,
    )
except ImportError as exc:
    print(exc)
    pytest.skip("onnx is an optional dependency", allow_module_level=True)


def test_visit_value_info():
    # initialize context
    ctx = OnnxXdslMapping()

    # create ValueInfoProto input tensor
    input_value_info = ValueInfoProto()
    input_value_info.name = "input_tensor"

    # define the type of the input tensor
    input_tensor_type = input_value_info.type.tensor_type
    input_tensor_type.elem_type = TensorProto.FLOAT
    input_tensor_type.shape.dim.extend(
        [
            onnx.TensorShapeProto.Dimension(dim_value=1),
            onnx.TensorShapeProto.Dimension(dim_value=3),
            onnx.TensorShapeProto.Dimension(dim_value=224),
            onnx.TensorShapeProto.Dimension(dim_value=224),
        ]
    )

    # run visit_value_info with empty context
    visit_value_info(input_value_info, ctx)

    # check keys in context
    keys = list(ctx.type_by_name.keys())
    assert keys == ["input_tensor"]

    # check type info
    type_info = str(ctx.type_by_name["input_tensor"])
    assert type_info == "tensor<1x3x224x224xf32>"


def test_visit_value_info_multiple_time():
    # initialize context
    ctx = OnnxXdslMapping()

    # create ValueInfoProto input tensor
    input_value_info = ValueInfoProto()
    input_value_info.name = "input_tensor"

    # define the type of the input tensor
    input_tensor_type = input_value_info.type.tensor_type
    input_tensor_type.elem_type = TensorProto.FLOAT
    input_tensor_type.shape.dim.extend(
        [
            onnx.TensorShapeProto.Dimension(dim_value=1),
            onnx.TensorShapeProto.Dimension(dim_value=3),
            onnx.TensorShapeProto.Dimension(dim_value=224),
            onnx.TensorShapeProto.Dimension(dim_value=224),
        ]
    )

    # run visit_value_info with empty context
    t1 = visit_value_info(input_value_info, ctx)

    # check type info
    assert isinstance(t1, Attribute)
    assert str(t1) == "tensor<1x3x224x224xf32>"

    # check keys in context
    keys = list(ctx.type_by_name.keys())
    assert keys == ["input_tensor"]

    # run visit_value_info again
    t2 = visit_value_info(input_value_info, ctx)

    # check type info
    assert isinstance(t2, Attribute)
    assert str(t2) == "tensor<1x3x224x224xf32>"

    # check keys in context
    keys = list(ctx.type_by_name.keys())
    assert keys == ["input_tensor"]

    # check it is returned the same reference
    assert t1 is t2
