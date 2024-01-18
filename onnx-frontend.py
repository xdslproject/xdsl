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
    from xdsl.builder import Builder
    from xdsl.ir import Attribute, SSAValue
    return Attribute, Builder, SSAValue


@app.cell
def __(
    Add,
    Attribute,
    Builder,
    FuncOp,
    GraphProto,
    ImplicitBuilder,
    ModuleOp,
    Return,
    SSA,
    SSAValue,
    Sequence,
    TensorType,
    ValueInfoProto,
    f32,
    model_def,
    onnx,
):
    from typing import TypeAlias, Any

    class Ctx:
        type_by_name: dict[str, Attribute]
        value_by_name: dict[str, SSA]

        def __init__(self):
            self.type_by_name = {}
            self.value_by_name = {}


    ELEM_TYPE = {
        1: f32,
    }

    def get_elem_type(code: int) -> Attribute:
        if code in ELEM_TYPE:
            return ELEM_TYPE[code]
        else:
            raise ValueError(f"Unknown elem_type: {code}")

    def get_shape(s: onnx.TensorShapeProto) -> tuple[int, ...]:
        return tuple(dim.dim_value for dim in s.dim)

    def get_tensor_type(t: onnx.TensorProto) -> TensorType[Any]:
        elem_type = get_elem_type(t.elem_type)
        shape = get_shape(t.shape)
        return TensorType(elem_type, shape)

    def _get_type(t: onnx.TypeProto) -> Attribute:
        tt = get_tensor_type(t.tensor_type)
        return tt

    def _visit_value_info(i: ValueInfoProto, ctx: Ctx) -> Attribute:
        name = i.name
        t = _get_type(i.type)
        ctx.type_by_name[name] = t
        return t

    OP_BY_OP_TYPE = {
        "Add": Add,
    }

    def visit_node(node: onnx.NodeProto, ctx: Ctx) -> Sequence[SSAValue]:
        if node.op_type not in OP_BY_OP_TYPE:
            raise ValueError(f"Unknown ONNX op name {node.op_type}")

        op = OP_BY_OP_TYPE[node.op_type]

        operands = tuple(ctx.value_by_name[name] for name in node.input)
        result_types = tuple(ctx.type_by_name[name] for name in node.output)

        return op.build(operands=operands, result_types=result_types).results

    def _visit_graph(b: Builder, g: GraphProto, ctx: Ctx) -> None:
        name = g.name

        input_types = tuple(_visit_value_info(input, ctx) for input in g.input)
        output_types = tuple(_visit_value_info(output, ctx) for output in g.output)

        func = FuncOp(name, (input_types, output_types))

        with ImplicitBuilder(func.body) as args:
            for input, arg in zip(g.input, args, strict=True):
                ctx.value_by_name[input.name] = arg

            for node in g.node:
                results = visit_node(node, ctx)
                for output_name, result in zip(node.output, results, strict=True):
                    ctx.value_by_name[output_name] = result

            returned_values = tuple(ctx.value_by_name[output.name] for output in g.output)
            Return(*returned_values)
            
        b.insert(func)


    def build_module(g: GraphProto) -> ModuleOp:
        module = ModuleOp([])
        b = Builder.at_start(module.body.block)
        ctx = Ctx()
        _visit_graph(b, g, ctx)

        print([(k, str(v)) for k, v in ctx.type_by_name.items()])

        return module


    module = build_module(model_def.graph)
    str(module)
    return (
        Any,
        Ctx,
        ELEM_TYPE,
        OP_BY_OP_TYPE,
        TypeAlias,
        build_module,
        get_elem_type,
        get_shape,
        get_tensor_type,
        module,
        visit_node,
    )


if __name__ == "__main__":
    app.run()
