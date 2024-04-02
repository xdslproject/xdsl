import pytest
from onnx import TensorProto, helper

try:
    from xdsl.frontend.onnx.context import (
        Ctx,  # noqa: E402
        visit_graph,  # noqa: E402
        visit_node,  # noqa: E402
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

    # define input and output names
    input1_name = "input1"
    input2_name = "input2"
    output_name = "output"

    # define Add node
    add_node = helper.make_node(
        op_type="Add",  # Operation type, addition
        inputs=[input1_name, input2_name],  # Input names
        outputs=[output_name],  # Output name
    )

    # create graph (composed of just one Add operation)
    graph = helper.make_graph(
        nodes=[add_node],
        name="add_graph",
        inputs=[
            helper.make_tensor_value_info(input1_name, TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info(input2_name, TensorProto.FLOAT, [None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None]),
        ],
    )

    visit_graph(graph, ctx)

    keys = list(ctx.value_by_name.keys())
    assert keys == ["input1", "input2", "output"]

    gen_ir = ctx.value_by_name[keys[2]].owner
    print(gen_ir)
    assert (
        str(gen_ir)
        == "%0 = onnx.Add(%1, %2) : (tensor<0x0xf32>, tensor<0x0xf32>) -> tensor<0x0xf32>"
    )
