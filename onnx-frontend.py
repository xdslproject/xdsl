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
    import numpy as np
    return np,


@app.cell
def __():
    import onnxruntime
    return onnxruntime,


@app.cell
def __(model_def, np, onnxruntime):
    np.random.seed(42)

    input1 = np.random.rand(3, 2).astype(np.float32)
    input2 = np.random.rand(3, 2).astype(np.float32)

    # Load the ONNX model
    session = onnxruntime.InferenceSession(model_def.SerializeToString())

    # Create a dictionary to hold the input data
    input_dict = {
        session.get_inputs()[0].name: input1,
        session.get_inputs()[1].name: input2
    }

    # Run the inference
    output = session.run(None, input_dict)

    # Output will contain the result of the computation
    result = output[0]

    # Display the inputs and result
    print("Input 1:\n", input1)
    print("Input 2:\n", input2)
    print("Result:\n", result)
    return input1, input2, input_dict, output, result, session


@app.cell
def __(mo):
    mo.md("""
    ```
    module {
      func.func @main_graph(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x2xf32> {
        %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
        return %0 : tensor<3x2xf32>
      }
    }
    ```
    """)
    return


@app.cell
def __():
    from xdsl.dialects.onnx import Add
    from xdsl.dialects.builtin import ModuleOp, TensorType, Float32Type, f32
    from xdsl.dialects.func import FuncOp, Return

    from xdsl.builder import ImplicitBuilder

    def tt(shape: tuple[int, ...]) -> TensorType[Float32Type]:
        return TensorType(f32, shape)
    return (
        Add,
        Float32Type,
        FuncOp,
        ImplicitBuilder,
        ModuleOp,
        Return,
        TensorType,
        f32,
        tt,
    )


@app.cell
def __(Add, FuncOp, ImplicitBuilder, ModuleOp, Return, tt):
    target = ModuleOp([])

    tt32 = tt((3, 2))


    with ImplicitBuilder(target.body):
        func = FuncOp("main_graph", ((tt32, tt32), (tt32,)))
        with ImplicitBuilder(func.body) as (arg0, arg1):
            res = Add(arg0, arg1).res
            Return(res)

    str(target)
    return arg0, arg1, func, res, target, tt32


@app.cell
def __():
    from xdsl.ir import Attribute, SSAValue
    return Attribute, SSAValue


@app.cell
def __(model_def):
    from xdsl.frontend.onnx import build_module as bm

    str(bm(model_def.graph))
    return bm,


if __name__ == "__main__":
    app.run()
