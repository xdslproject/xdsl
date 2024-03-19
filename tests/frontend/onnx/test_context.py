import pytest
from onnx import helper

try:
    from xdsl.frontend.onnx.context import (
        Ctx,  # noqa: E402
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
