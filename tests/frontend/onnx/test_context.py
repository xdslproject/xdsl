import onnx
import pytest

try:
    from onnx import (
        TensorProto,
        ValueInfoProto,
        helper,
    )

    from xdsl.frontend.onnx.context import (
        Ctx,  # noqa: E402
        build_module,  # noqa: E402
        visit_graph,  # noqa: E402
        visit_node,  # noqa: E402
        visit_value_info,  # noqa: E402
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
    ctx = Ctx()
    with pytest.raises(ValueError, match="Unknown ONNX op name dummy_op"):
        visit_node(node=node, ctx=ctx)


def test_visit_node_add():
    # initialize context
    ctx = Ctx()

    # create graph composed only of one Add operation
    graph, add_node = _create_graph_binary_op("Add", "add_graph")

    # visit graph
    visit_graph(graph, ctx)

    # visit node (test passes if no exceptions are raised)
    visit_node(add_node, ctx)


def test_visit_node_sub():
    # initialize context
    ctx = Ctx()

    # create graph composed only of one Sub operation
    graph, sub_node = _create_graph_binary_op("Sub", "sub_graph")

    # visit graph
    visit_graph(graph, ctx)

    # visit node (test passes if no exceptions are raised)
    visit_node(sub_node, ctx)


def test_visit_graph_add():
    # initialize context
    ctx = Ctx()

    # create graph composed only of one Add operation
    graph, _ = _create_graph_binary_op("Add", "add_graph")

    # visit graph
    visit_graph(graph, ctx)

    # check value_by_name keys
    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    # check expected generated ir
    gen_ir = ctx.value_by_name[keys[2]].owner
    print(gen_ir)
    assert (
        str(gen_ir)
        == "%0 = onnx.Add(%1, %2) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>"
    )


def test_visit_graph_sub():
    # initialize context
    ctx = Ctx()

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
    expected = (
        "builtin.module {\n"
        + "  func.func @add_graph(%0 : tensor<0x0xf32>, %1 : tensor<0x0xf32>) -> tensor<0x0xf32> {\n"
        + "    %2 = onnx.Add(%0, %1) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>\n"
        + "    func.return %2 : tensor<0x0xf32>\n"
        + "  }\n"
        + "}"
    )

    # check output
    assert str(module) == expected


def test_visit_value_info():
    # initialize context
    ctx = Ctx()
    print(ctx.type_by_name.keys())

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
