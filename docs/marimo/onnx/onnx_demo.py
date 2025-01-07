import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # ONNX to Snitch

        This notebook uses Marimo, a Jupyter-like notebook with interactive UI elements and reactive state.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    rank = mo.ui.slider(1, 4, value=2, label="Rank")

    mo.md(
        f"""
        For example, here is a slider, which can take on values from 1 to 4.

        {rank}
        """
    )
    return (rank,)


@app.cell(hide_code=True)
def _(mo, rank):
    shape = tuple(range(2, 2 + rank.value))

    mo.md(
        f"""
        We use the slider to determine the shape of our inputs and outputs:

        ```
        A: {'x'.join(str(dim) for dim in shape)}xf64
        B: {'x'.join(str(dim) for dim in shape)}xf64
        C: {'x'.join(str(dim) for dim in shape)}xf64
        ```
        """
    )
    return (shape,)


@app.cell(hide_code=True)
def _(mo, shape):
    mo.md(
        f"""
        ### The ONNX model

        We use the ONNX API to build a simple function, one that returns the elementwise sum of two arrays of shape {shape}
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import onnx
    from onnx import AttributeProto, GraphProto, TensorProto, ValueInfoProto, helper
    return (
        AttributeProto,
        GraphProto,
        TensorProto,
        ValueInfoProto,
        helper,
        onnx,
    )


@app.cell
def _(TensorProto, helper, onnx, shape):
    # Create one input (ValueInfoProto)
    X1 = helper.make_tensor_value_info("X1", TensorProto.DOUBLE, shape)
    X2 = helper.make_tensor_value_info("X2", TensorProto.DOUBLE, shape)

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, shape)

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        "Sub",  # node name
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
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=opset_import
    )

    onnx.checker.check_model(model_def)
    return X1, X2, Y, graph_def, model_def, node_def, opset_import


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ONNX uses a serialized binary format for neural networks, but can also print a string format, which can be useful for debugging.
        Here is the textual format of our model:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, model_def):
    mo.accordion(
        {
            "ONNX Graph": mo.plain_text(f"{model_def}"),
        }
    )
    return


@app.cell
def _(init_module, mo, xmo):
    mo.md(f"""
    ### Converting to `linalg`

    Here is the xDSL representation of the function, it takes two `tensor` values of our chosen shape, passes them as operands to the `onnx.Add` operation, and returns it:

    {xmo.module_html(init_module)}
    """
    )
    return


@app.cell
def _(build_module, model_def):
    init_module = build_module(model_def.graph)
    return (init_module,)


@app.cell(hide_code=True)
def _(MLContext, get_all_dialects):
    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx, dialect_factory, dialect_name


@app.cell
def _(mo):
    mo.md("""xDSL seamlessly interoperates with MLIR, we the `mlir-opt` tool to compile the input to a form that we want to process:""")
    return


@app.cell
def _(
    ConvertOnnxToLinalgPass,
    EmptyTensorToAllocTensorPass,
    MLIROptPass,
    init_module,
    mo,
    pipeline_accordion,
):
    bufferized_module, linalg_accordion = pipeline_accordion(
        (
            (
                mo.md(
                    """\
    We can use a pass implemented in xDSL to convert the ONNX operations to builtin operations, here we can use the `tensor.empty` op to create our output buffer, and `linalg.add` to represent the addition in destination-passing style:
    """
                ),
                ConvertOnnxToLinalgPass()
            ),
            (
                mo.md(
                    """
    We can also call into MLIR, here to convert `linalg.add` to `linalg.generic`, a representation of Einstein summation:
    """
                ),
                MLIROptPass(
                    generic=False,
                    arguments=["--linalg-generalize-named-ops"]
                )
            ),
            (
                mo.md(
                    """We prepare the result tensors for bufferization:"""
                ),
                EmptyTensorToAllocTensorPass()
            ),
            (
                mo.md(
                    """We then use MLIR to bufferize our function:"""
                ),
                MLIROptPass(
                    arguments=[
                        "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
                    ]
                )
            )
        ),
        init_module
    )

    linalg_accordion
    return bufferized_module, linalg_accordion


@app.cell
def _(mo):
    mo.md(
        """
        From here we can use a number of backends to generate executable code, like LLVM, or RISC-V assembly directly.
        Please see other notebooks for details
        """
    )
    return


@app.cell
def _():
    from xdsl.context import MLContext
    from xdsl.frontend.onnx.ir_builder import build_module
    from xdsl.ir import Attribute, SSAValue
    from xdsl.passes import PipelinePass
    from xdsl.tools.command_line_tool import get_all_dialects
    from xdsl.transforms.convert_onnx_to_linalg import ConvertOnnxToLinalgPass
    from xdsl.transforms.empty_tensor_to_alloc_tensor import EmptyTensorToAllocTensorPass
    from xdsl.transforms.mlir_opt import MLIROptPass
    return (
        Attribute,
        ConvertOnnxToLinalgPass,
        EmptyTensorToAllocTensorPass,
        MLContext,
        MLIROptPass,
        PipelinePass,
        SSAValue,
        build_module,
        get_all_dialects,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import xdsl.utils.marimo as xmo
    return (xmo,)


@app.cell(hide_code=True)
def _():
    from collections import Counter
    return (Counter,)


@app.cell(hide_code=True)
def _(Counter, ModuleOp, ModulePass, PipelinePass, ctx, mo, xmo):
    def spec_str(p: ModulePass) -> str:
        if isinstance(p, PipelinePass):
            return ",".join(str(c.pipeline_pass_spec()) for c in p.passes)
        else:
            return str(p.pipeline_pass_spec())

    def pipeline_accordion(passes: tuple[tuple[mo.Html, ModulePass], ...], module: ModuleOp) -> tuple[ModuleOp, mo.Html]:
        res = module.clone()
        d = {}
        total_key_count = Counter(spec_str(p) for _, p in passes)
        d_key_count = Counter()
        for text, p in passes:
            p.apply(ctx, res)
            spec = spec_str(p)
            d_key_count[spec] += 1
            if total_key_count[spec] != 1:
                header = f"{spec} ({d_key_count[spec]})"
            else:
                header = spec
            html_res = xmo.module_html(res)
            d[header] = mo.vstack((
                text,
                # mo.plain_text(f"Pass: {p.pipeline_pass_spec()}"),
                html_res
            ))
        return (res, mo.accordion(d))
    return pipeline_accordion, spec_str


if __name__ == "__main__":
    app.run()
