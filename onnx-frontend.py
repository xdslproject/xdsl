import marimo

__generated_with = "0.2.5"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    mo.md("""
    # ONNX to Snitch

    This notebook uses Marimo, a Jupyter-like notebook with interactive UI elements and reactive state. 
    """)
    return mo,


@app.cell
def __(mo):
    a = mo.ui.slider(1, 4, value=2)

    mo.md(f"""
    For example, here is a slider, which can take on values from 1 to 4.

    {a}
    """)
    return a,


@app.cell
def __(a, mo):
    shape = list(range(2, 2 + a.value))

    mo.md(f"""
    We use the slider to determine the shape of our inputs and outputs:

    {shape}
    """)
    return shape,


@app.cell
def __(mo, shape):
    import onnx
    from onnx import AttributeProto, GraphProto, TensorProto, ValueInfoProto, helper

    # Create one input (ValueInfoProto)
    X1 = helper.make_tensor_value_info("X1", TensorProto.DOUBLE, shape)
    X2 = helper.make_tensor_value_info("X2", TensorProto.DOUBLE, shape)

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, shape)

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
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=opset_import
    )

    print(f"The model is:\n{model_def}")
    onnx.checker.check_model(model_def)
    # onnx.save(model_def, "add.onnx")
    print("The model is checked!")

    mo.md(f"""
    ### The ONNX model

    We use the ONNX API to build a simple function, one that returns the elementwise sum of two arrays of shape {shape}
    """)
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
def __(mo):
    from xdsl.ir import Attribute, SSAValue

    mo.md("""
    We then convert the ONNX Graph to the xDSL representation, in the onnx dialect.
    """)
    return Attribute, SSAValue


@app.cell
def __(mo, model_def):
    from xdsl.frontend.onnx import build_module

    init_module = build_module(model_def.graph).clone()

    print(init_module)

    mo.md("""
    Here is the same function, it takes two `tensor` values of our chosen shape, passes them as operands to the `onnx.Add` operation, and returns it.
    """)
    return build_module, init_module


@app.cell
def __(init_module, mo):
    from xdsl.ir import MLContext
    from xdsl.tools.command_line_tool import get_all_dialects
    from xdsl.transforms.convert_onnx_to_linalg import ConvertOnnxToLinalgPass

    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    linalg_module = init_module.clone()

    ConvertOnnxToLinalgPass().apply(ctx, linalg_module)

    print(linalg_module)

    mo.md("""
    We can use a pass implemented in xDSL to convert the ONNX operations to builtin operations, here we can use the `tensor.empty` op to create our output buffer, and `linalg.add` to represent the addition in destination-passing style.
    """)
    return (
        ConvertOnnxToLinalgPass,
        MLContext,
        ctx,
        dialect_factory,
        dialect_name,
        get_all_dialects,
        linalg_module,
    )


@app.cell
def __(ctx, linalg_module, mo):
    from xdsl.transforms.mlir_opt import MLIROptPass

    generalized_module = linalg_module.clone()

    MLIROptPass(arguments=["--linalg-generalize-named-ops"]).apply(
        ctx, generalized_module
    )

    print(generalized_module)

    mo.md("""
    We can also call into MLIR, here to convert `linalg.add` to `linalg.generic`, a representation of Einstein summation.
    """)
    return MLIROptPass, generalized_module


@app.cell
def __(MLIROptPass, ctx, generalized_module, mo):
    bufferized_module = generalized_module.clone()

    MLIROptPass(
        arguments=[
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
        ]
    ).apply(ctx, bufferized_module)

    print(bufferized_module)

    mo.md("""
    We then use MLIR to bufferize our function.
    """)
    return bufferized_module,


@app.cell
def __(MLIROptPass, bufferized_module, ctx):
    scf_module = bufferized_module.clone()

    MLIROptPass(
        arguments=["--convert-linalg-to-loops", "--lower-affine", "--canonicalize"]
    ).apply(ctx, scf_module)

    print(scf_module)
    return scf_module,


@app.cell
def __(ctx, scf_module):
    from xdsl.backend.riscv.lowering import (
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
    )
    from xdsl.passes import PipelinePass
    from xdsl.transforms import reconcile_unrealized_casts

    riscv_module = scf_module.clone()

    lower_to_riscv = PipelinePass(
        [
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
        ]
    ).apply(ctx, riscv_module)

    print(riscv_module)
    return (
        PipelinePass,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
        lower_to_riscv,
        reconcile_unrealized_casts,
        riscv_module,
    )


@app.cell
def __(PipelinePass, ctx, riscv_module):
    from xdsl.backend.riscv.lowering.convert_snitch_stream_to_snitch import (
        ConvertSnitchStreamToSnitch,
    )
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.lower_snitch import LowerSnitchPass
    from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
    from xdsl.transforms.riscv_scf_loop_range_folding import (
        RiscvScfLoopRangeFoldingPass,
    )
    from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation

    regalloc_module = riscv_module.clone()

    PipelinePass(
        [
            RISCVRegisterAllocation(),
            CanonicalizePass(),
        ]
    ).apply(ctx, regalloc_module)

    print(regalloc_module)
    return (
        CanonicalizePass,
        ConvertSnitchStreamToSnitch,
        LowerSnitchPass,
        RISCVRegisterAllocation,
        RiscvScfLoopRangeFoldingPass,
        SnitchRegisterAllocation,
        regalloc_module,
    )


@app.cell
def __(CanonicalizePass, ctx, regalloc_module):
    from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
        ConvertRiscvScfToRiscvCfPass,
    )
    from xdsl.dialects.riscv import riscv_code

    assembly_module = regalloc_module.clone()

    ConvertRiscvScfToRiscvCfPass().apply(ctx, assembly_module)
    CanonicalizePass().apply(ctx, assembly_module)

    str(assembly_module)
    return ConvertRiscvScfToRiscvCfPass, assembly_module, riscv_code


@app.cell
def __(assembly_module, mo, riscv_code):
    assembly = riscv_code(assembly_module)

    assembly

    mo.md("""
    This representation of the program in xDSL corresponds ~1:1 to RISC-V assembly, and we can use a helper function to print that out.
    """)
    return assembly,


@app.cell
def __(shape):
    from math import prod

    from xdsl.interpreters.riscv import RawPtr

    n = prod(shape)

    lhs = RawPtr.new_float64([i + 1 for i in range(n)])
    rhs = RawPtr.new_float64([(i + 1) / 100 for i in range(n)])

    lhs.float64.get_list(n), rhs.float64.get_list(n)
    return RawPtr, lhs, n, prod, rhs


@app.cell
def __(a, ctx, lhs, mo, n, rhs, riscv_module, shape):
    from xdsl.interpreter import Interpreter
    from xdsl.interpreters import register_implementations

    interpreter = Interpreter(riscv_module)

    register_implementations(interpreter, ctx, include_wgpu=False)

    (res,) = interpreter.call_op("main_graph", (lhs, rhs))

    res.float64.get_list(n)

    mo.md(f"""
    One of the useful features of xDSL is its interpreter. Here we've implemented all the necessary functions to interpret the code at a low level, to check that our compilation is correct. Here's the slider modifying the shape variable defined above, we can slide it to see the result of the code compiled with different input shapes, and interpreted at the RISC-V level.

    Rank: {a}
    Shape: {shape}

    ```
    LHS:    {lhs.float64.get_list(n)}
    RHS:    {rhs.float64.get_list(n)}
    Result: {res.float64.get_list(n)}
    ```
    """)
    return Interpreter, interpreter, register_implementations, res


if __name__ == "__main__":
    app.run()
