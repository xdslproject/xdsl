import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import onnx
    from onnx import helper
    from onnx import AttributeProto, TensorProto, GraphProto, ValueInfoProto

    # Create one input (ValueInfoProto)
    X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [3, 2])
    X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [3, 2])

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        "Add",  # node name
        ["X1", "X2"],  # inputs
        ["Y"],  # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "main_graph",
        [X1, X2],
        [Y],
    )


    # Set opset version to 18
    opset_import = [helper.make_operatorsetid("", 18)]

    # Create the model (ModelProto) without using helper.make_model
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=opset_import)

    print("The model is:\n{}".format(model_def))
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "add.onnx")
    print("The model is checked!")
    return (
        AttributeProto,
        GraphProto,
        TensorProto,
        ValueInfoProto,
        X1,
        X2,
        Y,
        graph_def,
        helper,
        model_def,
        node_def,
        onnx,
        opset_import,
    )


@app.cell
def __():
    from xdsl.ir import Attribute, SSAValue
    return Attribute, SSAValue


@app.cell
def __(model_def):
    from xdsl.frontend.onnx import build_module

    module = build_module(model_def.graph)

    str(module)
    return build_module, module


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
