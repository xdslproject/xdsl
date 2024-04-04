import onnx
import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func

try:
    from onnx import GraphProto, TensorProto, ValueInfoProto, helper

    from xdsl.frontend.onnx.context import (
        OnnxXdslMapping,
        build_module,
        visit_graph,
        visit_node,
        visit_value_info,
    )
except ImportError as exc:
    print(exc)
    pytest.skip("onnx is an optional dependency", allow_module_level=True)


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

    # create graph composed only of one Sub operation
    graph, sub_node = _create_graph_binary_op("Add", "add_graph")

    # update context
    _update_context(graph, ctx)

    # expected output before visinting the op node
    expected_output_pre = """{'input1': <BlockArgument[tensor<0x0xf32>] index: 0, uses: 0>, 'input2': <BlockArgument[tensor<0x0xf32>] index: 1, uses: 0>}"""

    assert str(ctx.value_by_name) == expected_output_pre

    # visit node
    visit_node(sub_node, ctx)

    # expected output after visinting the op node
    expected_output_post = """{'input1': <BlockArgument[tensor<0x0xf32>] index: 0, uses: 1>, 'input2': <BlockArgument[tensor<0x0xf32>] index: 1, uses: 1>, 'output': <OpResult[tensor<0x0xf32>] index: 0, operation: onnx.Add, uses: 0>}"""

    assert str(ctx.value_by_name) == expected_output_post


def test_visit_node_sub():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Sub operation
    graph, sub_node = _create_graph_binary_op("Sub", "sub_graph")

    # update context
    _update_context(graph, ctx)

    # expected output before visinting the op node
    expected_output_pre = """{'input1': <BlockArgument[tensor<0x0xf32>] index: 0, uses: 0>, 'input2': <BlockArgument[tensor<0x0xf32>] index: 1, uses: 0>}"""

    assert str(ctx.value_by_name) == expected_output_pre

    # visit node
    visit_node(sub_node, ctx)

    # expected output after visinting the op node
    expected_output_post = """{'input1': <BlockArgument[tensor<0x0xf32>] index: 0, uses: 1>, 'input2': <BlockArgument[tensor<0x0xf32>] index: 1, uses: 1>, 'output': <OpResult[tensor<0x0xf32>] index: 0, operation: onnx.Sub, uses: 0>}"""

    assert str(ctx.value_by_name) == expected_output_post


def test_visit_graph_add():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Add operation
    graph, _ = _create_graph_binary_op("Add", "add_graph")

    # visit graph
    visit_graph(graph, ctx)

    # check value_by_name keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    # check expected generated ir
    gen_ir = ctx.value_by_name[keys[2]].owner
    assert (
        str(gen_ir)
        == "%0 = onnx.Add(%1, %2) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>"
    )


def test_visit_graph_sub():
    # initialize context
    ctx = OnnxXdslMapping()

    # create graph composed only of one Sub operation
    graph, _ = _create_graph_binary_op("Sub", "sub_graph")

    # run visit graph
    visit_graph(graph, ctx)

    # check value_by_names keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    # check generated ir
    gen_ir = ctx.value_by_name[keys[2]].owner
    assert (
        str(gen_ir)
        == "%0 = onnx.Sub(%1, %2) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>"
    )


def test_build_module():
    # create graph composed only of one Add operation
    graph, _ = _create_graph_binary_op("Add", "add_graph")

    # create module
    module = build_module(graph)

    # define expected output
    expected = """
builtin.module {
  func.func @add_graph(%0 : tensor<0x0xf32>, %1 : tensor<0x0xf32>) -> tensor<0x0xf32> {
    %2 = onnx.Add(%0, %1) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>
    func.return %2 : tensor<0x0xf32>
  }
}"""

    # remove first new line
    expected = expected[1:]

    # check output
    assert str(module) == expected


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


def test_addition_model():
    # load the onnx model
    model = onnx.load("tests/frontend/onnx/addition_model.onnx")

    # access the graph
    graph = model.graph

    # expected output
    expected_output = """
builtin.module {
  func.func @add_graph(%0 : tensor<0x0xf32>, %1 : tensor<0x0xf32>) -> tensor<0x0xf32> {
    %2 = onnx.Add(%0, %1) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>
    func.return %2 : tensor<0x0xf32>
  }
}"""
    expected_output = expected_output[1:]

    # module
    module = build_module(graph)

    assert str(module) == expected_output


def _create_graph_binary_op(op_name: str, graph_name: str):
    # define input and output names
    input1_name = "input1"
    input2_name = "input2"
    output_name = "output"

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
            helper.make_tensor_value_info(input1_name, TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info(input2_name, TensorProto.FLOAT, [None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None]),
        ],
    )

    return graph, op_node


def _update_context(graph: GraphProto, ctx: OnnxXdslMapping):
    name = graph.name
    input_types = tuple(visit_value_info(input, ctx) for input in graph.input)
    output_types = tuple(visit_value_info(output, ctx) for output in graph.output)

    fn = func.FuncOp(name, (input_types, output_types))
    with ImplicitBuilder(fn.body) as args:
        for input, arg in zip(graph.input, args, strict=True):
            ctx.value_by_name[input.name] = arg
