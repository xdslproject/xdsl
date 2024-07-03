import marimo

__generated_with = "0.6.25"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    mo.md(
        """
    # ONNX to Snitch

    This notebook uses Marimo, a Jupyter-like notebook with interactive UI elements and reactive state.
    """
    )
    return mo,


@app.cell
def __(mo):
    rank = mo.ui.slider(1, 4, value=2, label="Rank")

    mo.md(
        f"""
    For example, here is a slider, which can take on values from 1 to 4.

    {rank}
    """
    )
    return rank,


@app.cell
def __(mo, rank):
    shape = tuple(range(2, 2 + rank.value))

    mo.md(
        f"""
    We use the slider to determine the shape of our inputs and outputs:

    {shape}
    """
    )
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

    print(f"The model is:\n{model_def}")
    onnx.checker.check_model(model_def)
    # onnx.save(model_def, "add.onnx")
    print("The model is checked!")

    mo.md(
        f"""
    ### The ONNX model

    We use the ONNX API to build a simple function, one that returns the elementwise sum of two arrays of shape {shape}
    """
    )
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
    # As of version 0.6.14, something breaks when importing from xDSL in multiple cells
    # https://github.com/marimo-team/marimo/issues/1699

    from xdsl.backend.riscv.lowering import (
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
    )
    from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
        ConvertRiscvScfToRiscvCfPass,
    )
    from xdsl.backend.riscv.lowering.convert_snitch_stream_to_snitch import (
        ConvertSnitchStreamToSnitch,
    )
    from xdsl.context import MLContext
    from xdsl.dialects.riscv import riscv_code
    from xdsl.frontend.onnx.ir_builder import build_module
    from xdsl.interpreter import Interpreter, OpCounter
    from xdsl.interpreters import register_implementations
    from xdsl.interpreters.ptr import TypedPtr
    from xdsl.ir import Attribute, SSAValue
    from xdsl.passes import PipelinePass
    from xdsl.tools.command_line_tool import get_all_dialects
    from xdsl.transforms import (
        arith_add_fastmath,
        convert_linalg_to_memref_stream,
        convert_memref_stream_to_loops,
        convert_memref_stream_to_snitch_stream,
        convert_riscv_scf_for_to_frep,
        dead_code_elimination,
        loop_hoist_memref,
        lower_affine,
        memref_streamify,
        reconcile_unrealized_casts,
        test_lower_snitch_stream_to_asm,
    )
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.convert_onnx_to_linalg import ConvertOnnxToLinalgPass
    from xdsl.transforms.lower_snitch import LowerSnitchPass
    from xdsl.transforms.mlir_opt import MLIROptPass
    from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
    from xdsl.transforms.riscv_scf_loop_range_folding import (
        RiscvScfLoopRangeFoldingPass,
    )
    from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation

    mo.md(
        """
    We then convert the ONNX Graph to the xDSL representation, in the onnx dialect.
    """
    )
    return (
        Attribute,
        CanonicalizePass,
        ConvertOnnxToLinalgPass,
        ConvertRiscvScfToRiscvCfPass,
        ConvertSnitchStreamToSnitch,
        Interpreter,
        LowerSnitchPass,
        MLContext,
        MLIROptPass,
        OpCounter,
        PipelinePass,
        RISCVRegisterAllocation,
        RiscvScfLoopRangeFoldingPass,
        SSAValue,
        SnitchRegisterAllocation,
        TypedPtr,
        arith_add_fastmath,
        build_module,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_linalg_to_memref_stream,
        convert_memref_stream_to_loops,
        convert_memref_stream_to_snitch_stream,
        convert_memref_to_riscv,
        convert_riscv_scf_for_to_frep,
        convert_scf_to_riscv_scf,
        dead_code_elimination,
        get_all_dialects,
        loop_hoist_memref,
        lower_affine,
        memref_streamify,
        reconcile_unrealized_casts,
        register_implementations,
        riscv_code,
        test_lower_snitch_stream_to_asm,
    )


@app.cell
def __(build_module, mo, model_def):
    init_module = build_module(model_def.graph).clone()

    print(init_module)

    mo.md(
        """
    Here is the same function, it takes two `tensor` values of our chosen shape, passes them as operands to the `onnx.Add` operation, and returns it.
    """
    )
    return init_module,


@app.cell
def __(
    ConvertOnnxToLinalgPass,
    MLContext,
    get_all_dialects,
    init_module,
    mo,
):
    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    linalg_module = init_module.clone()

    ConvertOnnxToLinalgPass().apply(ctx, linalg_module)

    print(linalg_module)

    mo.md(
        """
    We can use a pass implemented in xDSL to convert the ONNX operations to builtin operations, here we can use the `tensor.empty` op to create our output buffer, and `linalg.add` to represent the addition in destination-passing style.
    """
    )
    return ctx, dialect_factory, dialect_name, linalg_module


@app.cell
def __(MLIROptPass, ctx, linalg_module, mo):
    generalized_module = linalg_module.clone()

    MLIROptPass(generic=False, arguments=["--linalg-generalize-named-ops"]).apply(
        ctx, generalized_module
    )

    print(generalized_module)

    mo.md(
        """
    We can also call into MLIR, here to convert `linalg.add` to `linalg.generic`, a representation of Einstein summation.
    """
    )
    return generalized_module,


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

    mo.md(
        """
    We then use MLIR to bufferize our function.
    """
    )
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
def __(
    PipelinePass,
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_memref_to_riscv,
    convert_scf_to_riscv_scf,
    ctx,
    reconcile_unrealized_casts,
    scf_module,
):
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
    return lower_to_riscv, riscv_module


@app.cell
def __(
    CanonicalizePass,
    PipelinePass,
    RISCVRegisterAllocation,
    ctx,
    riscv_module,
):
    regalloc_module = riscv_module.clone()

    PipelinePass(
        [
            RISCVRegisterAllocation(),
            CanonicalizePass(),
        ]
    ).apply(ctx, regalloc_module)

    print(regalloc_module)
    return regalloc_module,


@app.cell
def __(
    CanonicalizePass,
    ConvertRiscvScfToRiscvCfPass,
    ctx,
    regalloc_module,
):
    assembly_module = regalloc_module.clone()

    ConvertRiscvScfToRiscvCfPass().apply(ctx, assembly_module)
    CanonicalizePass().apply(ctx, assembly_module)

    print(assembly_module)
    return assembly_module,


@app.cell
def __(assembly_module, mo, riscv_code):
    assembly = riscv_code(assembly_module)

    print(assembly)

    mo.md(
        """
    This representation of the program in xDSL corresponds ~1:1 to RISC-V assembly, and we can use a helper function to print that out.
    """
    )
    return assembly,


@app.cell
def __(
    CanonicalizePass,
    PipelinePass,
    arith_add_fastmath,
    bufferized_module,
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_linalg_to_memref_stream,
    convert_memref_stream_to_loops,
    convert_memref_stream_to_snitch_stream,
    convert_memref_to_riscv,
    convert_riscv_scf_for_to_frep,
    convert_scf_to_riscv_scf,
    ctx,
    dead_code_elimination,
    loop_hoist_memref,
    lower_affine,
    memref_streamify,
    mo,
    reconcile_unrealized_casts,
):
    snitch_stream_module = bufferized_module.clone()

    pass_pipeline = PipelinePass(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemrefStreamPass(),
            memref_streamify.MemrefStreamifyPass(),
            convert_memref_stream_to_loops.ConvertMemrefStreamToLoopsPass(),
            convert_memref_stream_to_snitch_stream.ConvertMemrefStreamToSnitch(),
            arith_add_fastmath.AddArithFastMathFlagsPass(),
            loop_hoist_memref.LoopHoistMemrefPass(),
            lower_affine.LowerAffinePass(),
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            CanonicalizePass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            dead_code_elimination.DeadCodeElimination(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
            convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
        ]
    )

    pass_pipeline.apply(ctx, snitch_stream_module)

    print(snitch_stream_module)

    mo.md(
        """
    ### Compiling to Snitch

    xDSL is also capable of targeting Snitch, and making use of its streaming registers and fixed-repetition loop. We use a different lowering flow from the linalg.generic representation to represent a high-level, structured, but Snitch-specific representation of the code:
    """
    )
    return pass_pipeline, snitch_stream_module


@app.cell
def __(
    ctx,
    mo,
    riscv_code,
    snitch_stream_module,
    test_lower_snitch_stream_to_asm,
):
    snitch_asm_module = snitch_stream_module.clone()

    test_lower_snitch_stream_to_asm.TestLowerSnitchStreamToAsm().apply(
        ctx, snitch_asm_module
    )

    print(riscv_code(snitch_asm_module))

    mo.md(
        """
    We can then lower this to assembly that includes assembly instructions from the Snitch-extended ISA:
    """
    )
    return snitch_asm_module,


@app.cell
def __(
    Interpreter,
    OpCounter,
    TypedPtr,
    ctx,
    mo,
    rank,
    register_implementations,
    riscv_module,
    shape,
):
    from math import prod

    n = prod(shape)

    lhs = TypedPtr.new_float64([i + 1 for i in range(n)]).raw
    rhs = TypedPtr.new_float64([(i + 1) / 100 for i in range(n)]).raw

    lhs.float64.get_list(n), rhs.float64.get_list(n)

    riscv_op_counter = OpCounter()
    riscv_interpreter = Interpreter(riscv_module, listener=riscv_op_counter)

    register_implementations(riscv_interpreter, ctx, include_wgpu=False)

    (riscv_res,) = riscv_interpreter.call_op("main_graph", (lhs, rhs))

    mo.md(
        f"""
    One of the useful features of xDSL is its interpreter. Here we've implemented all the necessary functions to interpret the code at a low level, to check that our compilation is correct. Here's the slider modifying the shape variable defined above, we can slide it to see the result of the code compiled with different input shapes, and interpreted at the RISC-V level.

    {rank}

    ```
    Shape:  {shape}
    LHS:    {lhs.float64.get_list(n)}
    RHS:    {rhs.float64.get_list(n)}
    Result: {riscv_res.float64.get_list(n)}
    ```
    """
    )
    return lhs, n, prod, rhs, riscv_interpreter, riscv_op_counter, riscv_res


@app.cell
def __(
    Interpreter,
    OpCounter,
    ctx,
    lhs,
    mo,
    n,
    register_implementations,
    rhs,
    snitch_stream_module,
):
    snitch_op_counter = OpCounter()
    snitch_interpreter = Interpreter(snitch_stream_module, listener=snitch_op_counter)

    register_implementations(snitch_interpreter, ctx, include_wgpu=False)

    (snitch_res,) = snitch_interpreter.call_op("main_graph", (lhs, rhs))

    mo.md(
        f"""
    We can also interpret the Snitch version of the code to compare the results:

    Snitch result: {snitch_res.float64.get_list(n)}
    """
    )
    return snitch_interpreter, snitch_op_counter, snitch_res


@app.cell
def __(mo, rank, riscv_op_counter, snitch_op_counter):
    rv_dict = dict(riscv_op_counter.ops)
    sn_dict = dict(snitch_op_counter.ops)

    all_keys = sorted(set(rv_dict) | set(sn_dict))
    max_len_key = max(len(k) for k in all_keys)
    max_len_value = max(len(str(val)) for val in (*rv_dict.values(), *sn_dict.values()))

    def format_row(key: str, *values: str):
        paddings = tuple(" " * (max_len_value - len(val)) for val in values)
        vals = "".join(f"\t{padding}{val}" for padding, val in zip(paddings, values))
        return f"{key}{' ' * (max_len_key - len(key))}{vals}\n"

    rows = " " * max_len_key + "\trv\tsn\tdiff\n"

    ZERO_VAL = "."

    for key in all_keys:
        rv_val = rv_dict.get(key, 0)
        sn_val = sn_dict.get(key, 0)
        diff_val = sn_val - rv_val

        rv_str = str(rv_val) if rv_val else ZERO_VAL
        sn_str = str(sn_val) if sn_val else ZERO_VAL
        diff_str = (
            (f"+{diff_val}" if diff_val > 0 else f"{diff_val}") if diff_val else "="
        )

        rows += key
        rows += " " * (max_len_key - len(key))
        rows += "\t"
        rows += " " * (max_len_value - len(rv_str))
        rows += rv_str
        rows += "\t"
        rows += " " * (max_len_value - len(sn_str))
        rows += sn_str
        rows += "\t"
        rows += " " * (max_len_value - len(diff_str))
        rows += diff_str
        rows += "\n"

    rv_sum = sum(rv_dict.values())
    sn_sum = sum(sn_dict.values())
    total_diff = sn_sum - rv_sum

    rows += format_row("total", str(rv_sum), str(sn_sum), str(total_diff))

    mo.md(
        f"""
    We can also compare the number of instructions executed in our interpreter runs:

    {rank}

    ```
    {rows}
    ```
    """
    )
    return (
        ZERO_VAL,
        all_keys,
        diff_str,
        diff_val,
        format_row,
        key,
        max_len_key,
        max_len_value,
        rows,
        rv_dict,
        rv_str,
        rv_sum,
        rv_val,
        sn_dict,
        sn_str,
        sn_sum,
        sn_val,
        total_diff,
    )


if __name__ == "__main__":
    app.run()
