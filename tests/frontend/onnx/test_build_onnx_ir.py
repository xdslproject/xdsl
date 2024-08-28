import onnx
import pytest

from xdsl.dialects.builtin import TensorType, f32
from xdsl.dialects.onnx import Add, MatMul, Sub, Transpose
from xdsl.ir import Attribute
from xdsl.utils.test_value import TestSSAValue

try:
    from onnx import TensorProto, ValueInfoProto, helper

    from xdsl.frontend.onnx.ir_builder import (
        OnnxXdslMapping,
        build_module,
        visit_graph,
        visit_node,
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


def test_visit_node_unknown_op_name():
    """
    Test for unknown expected onnx op
    """
    node_attributes = {
        "name": "dummy_name",
        "op_type": "dummy_op",
    }

    node = helper.make_node(
        **node_attributes, inputs=["input1", "input2"], outputs=["output"]
    )
    ctx = OnnxXdslMapping()
    with pytest.raises(ValueError, match="Unknown ONNX op name dummy_op"):
        visit_node(node=node, ctx=ctx)


def test_visit_node_add():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Add operation
    _, add_node = _create_graph_binary_op("Add", "add_graph", [64], [64], [64])

    lhs = TestSSAValue(TensorType(f32, [64]))
    rhs = TestSSAValue(TensorType(f32, [64]))
    ctx.value_by_name["input1"] = lhs
    ctx.value_by_name["input2"] = rhs

    lhs_type = TensorType(f32, [64])
    rhs_type = TensorType(f32, [64])
    out_type = TensorType(f32, [64])
    ctx.type_by_name["input1"] = lhs_type
    ctx.type_by_name["input2"] = rhs_type
    ctx.type_by_name["output"] = out_type

    # visit node
    op = visit_node(add_node, ctx)

    assert isinstance(op, Add)
    assert op.lhs is lhs
    assert op.rhs is rhs
    assert op.res is ctx.value_by_name["output"]
    assert not op.attributes
    assert not op.regions


def test_visit_node_sub():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Sub operation
    _, sub_node = _create_graph_binary_op("Sub", "sub_graph", [64], [64], [64])

    lhs = TestSSAValue(TensorType(f32, [64]))
    rhs = TestSSAValue(TensorType(f32, [64]))
    ctx.value_by_name["input1"] = lhs
    ctx.value_by_name["input2"] = rhs

    lhs_type = TensorType(f32, [64])
    rhs_type = TensorType(f32, [64])
    out_type = TensorType(f32, [64])
    ctx.type_by_name["input1"] = lhs_type
    ctx.type_by_name["input2"] = rhs_type
    ctx.type_by_name["output"] = out_type

    op = visit_node(sub_node, ctx)

    assert isinstance(op, Sub)
    assert op.lhs is lhs
    assert op.rhs is rhs
    assert op.res is ctx.value_by_name["output"]
    assert not op.attributes
    assert not op.regions


def test_visit_node_matmul():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Sub operation
    _, matmul_node = _create_graph_binary_op(
        "MatMul", "matmul_graph", [64, 128], [128, 64], [64, 64]
    )

    lhs = TestSSAValue(TensorType(f32, [64, 128]))
    rhs = TestSSAValue(TensorType(f32, [128, 64]))
    ctx.value_by_name["input1"] = lhs
    ctx.value_by_name["input2"] = rhs

    lhs_type = TensorType(f32, [64, 128])
    rhs_type = TensorType(f32, [128, 64])
    out_type = TensorType(f32, [64, 64])
    ctx.type_by_name["input1"] = lhs_type
    ctx.type_by_name["input2"] = rhs_type
    ctx.type_by_name["output"] = out_type

    op = visit_node(matmul_node, ctx)
    op.verify()

    assert isinstance(op, MatMul)
    assert op.matrix_A is lhs
    assert op.matrix_B is rhs
    assert op.matrix_Y is ctx.value_by_name["output"]
    assert not op.attributes
    assert not op.regions


def test_visit_node_transpose():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Sub operation
    _, transpose_node = _create_transpose_op(
        graph_name="transpose_graph", dim_in=[64, 128], dim_out=[128, 64], perm=[1, 0]
    )

    in_value = TestSSAValue(TensorType(f32, [64, 128]))
    ctx.value_by_name["input"] = in_value

    in_type = TensorType(f32, [64, 128])
    out_type = TensorType(f32, [128, 64])
    ctx.type_by_name["input"] = in_type
    ctx.type_by_name["output"] = out_type

    op = visit_node(transpose_node, ctx)
    op.verify()

    assert isinstance(op, Transpose)
    assert op.tensor_input is in_value
    assert op.tensor_output is ctx.value_by_name["output"]
    assert not op.attributes
    assert not op.regions


def test_visit_graph_add():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Add operation
    graph, _ = _create_graph_binary_op("Add", "add_graph", [64], [64], [64])

    # visit graph
    visit_graph(graph, ctx)

    # check value_by_name keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    # check expected generated ir
    gen_ir = ctx.value_by_name[keys[2]].owner
    assert (
        str(gen_ir)
        == "%0 = onnx.Add(%1, %2) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>"
    )


def test_visit_graph_sub():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Sub operation
    graph, _ = _create_graph_binary_op("Sub", "sub_graph", [64], [64], [64])

    # run visit graph
    visit_graph(graph, ctx)

    # check value_by_names keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    # check generated ir
    gen_ir = ctx.value_by_name[keys[2]].owner
    assert (
        str(gen_ir)
        == "%0 = onnx.Sub(%1, %2) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>"
    )


def test_visit_graph_matmul():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one MatMul operation
    graph, _ = _create_graph_binary_op("MatMul", "matmul_graph", [64], [64], [64])

    # run visit graph
    visit_graph(graph, ctx)

    # check value_by_names keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    # check generated ir
    gen_ir = ctx.value_by_name[keys[2]].owner
    assert (
        str(gen_ir)
        == "%0 = onnx.MatMul(%1, %2) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>"
    )


def test_visit_graph_transpose():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Transpose operation
    graph, _ = _create_transpose_op("transpose_graph", [64, 128], [128, 64], [1, 0])

    # run visit graph
    visit_graph(graph, ctx)

    # check value_by_names keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input", "output"]

    # check generated ir
    gen_ir = ctx.value_by_name[keys[1]].owner
    assert (
        str(gen_ir)
        == "%0 = onnx.Transpose(%1) : (tensor<64x128xf32>) -> tensor<128x64xf32>"
    )


def test_build_module():
    # create graph composed only of one Add operation
    graph, _ = _create_graph_binary_op("Add", "add_graph", [64], [64], [64])

    # create module
    module = build_module(graph)

    # define expected output
    expected = """
builtin.module {
  func.func @add_graph(%0 : tensor<64xf32>, %1 : tensor<64xf32>) -> tensor<64xf32> {
    %2 = onnx.Add(%0, %1) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    func.return %2 : tensor<64xf32>
  }
}"""

    # remove first new line
    expected = expected[1:]

    # check output
    assert str(module) == expected


def _create_graph_binary_op(
    op_name: str,
    graph_name: str,
    dim_in1: list[int],
    dim_in2: list[int],
    dim_out: list[int],
):
    # define input and output names
    input1_name = "input1"
    input2_name = "input2"
    output_name = "output"

    # define input shapes
    input1_shape = dim_in1
    input2_shape = dim_in2
    output_shape = dim_out

    # define op node
    op_node = helper.make_node(
        op_type=op_name,
        inputs=[input1_name, input2_name],
        outputs=[output_name],
    )

    # create graph (composed of just one operation)
    graph = helper.make_graph(
        nodes=[op_node],
        name=graph_name,
        inputs=[
            helper.make_tensor_value_info(input1_name, TensorProto.FLOAT, input1_shape),
            helper.make_tensor_value_info(input2_name, TensorProto.FLOAT, input2_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape),
        ],
    )

    return graph, op_node


def _create_transpose_op(
    graph_name: str, dim_in: list[int], dim_out: list[int], perm: list[int]
):
    # create input tensor
    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, dim_in
    )

    # create output tensor
    output_tensor = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, dim_out
    )

    # create transpose operation
    transpose_node = helper.make_node("Transpose", ["input"], ["output"], perm=perm)

    # create onnx graph
    graph_def = helper.make_graph(
        [transpose_node], graph_name, [input_tensor], [output_tensor]
    )

    return graph_def, transpose_node
