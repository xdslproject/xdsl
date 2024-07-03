import marimo

__generated_with = "0.6.25"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # ONNX to Snitch

        This notebook uses Marimo, a Jupyter-like notebook with interactive UI elements and reactive state.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    rank = mo.ui.slider(1, 4, value=2, label="Rank")

    mo.md(
        f"""
        For example, here is a slider, which can take on values from 1 to 4.

        {rank}
        """
    )
    return rank,


@app.cell(hide_code=True)
def __(mo, rank):
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
    return shape,


@app.cell(hide_code=True)
def __(mo, shape):
    mo.md(
        f"""
        ### The ONNX model

        We use the ONNX API to build a simple function, one that returns the elementwise sum of two arrays of shape {shape}
        """
    )
    return


@app.cell(hide_code=True)
def __():
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
def __(TensorProto, helper, onnx, shape):
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
def __(mo):
    mo.md(
        """
        ONNX uses a serialized binary format for neural networks, but can also print a string format, which can be useful for debugging.
        Here is the textual format of our model:
        """
    )
    return


@app.cell(hide_code=True)
def __(mo, model_def):
    mo.accordion(
        {
            "ONNX Graph": mo.plain_text(f"{model_def}"),
        }
    )
    return


@app.cell(hide_code=True)
def __(html, init_module, mo):
    mo.md(f"""
    ### Compiling to RISC-V

    Here is the xDSL representation of the function, it takes two `tensor` values of our chosen shape, passes them as operands to the `onnx.Add` operation, and returns it:

    {html(init_module)}
    """
    )
    return


@app.cell
def __(build_module, model_def):
    init_module = build_module(model_def.graph)
    return init_module,


@app.cell(hide_code=True)
def __(MLContext, get_all_dialects):
    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx, dialect_factory, dialect_name


@app.cell(hide_code=True)
def __(mo):
    mo.md("xDSL seamlessly interoperates with MLIR, we the `mlir-opt` tool to compile the input to a form that we want to process:")
    return


@app.cell(hide_code=True)
def __(
    ConvertOnnxToLinalgPass,
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
                    """We then use MLIR to bufferize our function:"""
                ),
                MLIROptPass(
                    arguments=[
                        "--empty-tensor-to-alloc-tensor",
                        "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
                    ]
                )
            )
        ),
        init_module
    )

    linalg_accordion
    return bufferized_module, linalg_accordion


@app.cell(hide_code=True)
def __(mo):
    mo.md("We can take this representation, and lower to RISC-V-specific dialects:")
    return


@app.cell
def __(
    PipelinePass,
    bufferized_module,
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_linalg_to_loops,
    convert_memref_to_riscv,
    convert_scf_to_riscv_scf,
    pipeline_accordion,
    reconcile_unrealized_casts,
):
    lower_to_riscv = PipelinePass(
        [
            convert_linalg_to_loops.ConvertLinalgToLoopsPass(),
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
        ]
    )

    riscv_module, riscv_accordion = pipeline_accordion(
        tuple(("", p) for p in lower_to_riscv.passes), bufferized_module
    )

    riscv_accordion
    return lower_to_riscv, riscv_accordion, riscv_module


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        #### Register allocation

        We implemented a register allocator for our RISC-V representation, that works on functions with structured control flow:
        """
    )
    return


@app.cell
def __(
    CanonicalizePass,
    PipelinePass,
    RISCVRegisterAllocation,
    pipeline_accordion,
    riscv_module,
):
    allocate_registers = PipelinePass(
        [
            RISCVRegisterAllocation(),
            CanonicalizePass(),
        ]
    )

    regalloc_module, regalloc_accordion = pipeline_accordion(
        tuple(("", p) for p in allocate_registers.passes), riscv_module
    )

    regalloc_accordion
    return allocate_registers, regalloc_accordion, regalloc_module


@app.cell
def __(
    CanonicalizePass,
    ConvertRiscvScfToRiscvCfPass,
    PipelinePass,
    pipeline_accordion,
    regalloc_module,
):
    lower_to_asm = PipelinePass(
        [
            ConvertRiscvScfToRiscvCfPass(),
            CanonicalizePass(),
        ]
    )

    riscv_asm_module, assembly_accordion = pipeline_accordion(
        (("", lower_to_asm),), regalloc_module
    )

    assembly_accordion
    return assembly_accordion, lower_to_asm, riscv_asm_module


@app.cell(hide_code=True)
def __(mo):
    mo.md("This representation of the program in xDSL corresponds ~1:1 to RISC-V assembly, and we can use a helper function to print that out.")
    return


@app.cell(hide_code=True)
def __(mo, riscv_asm_module, riscv_code):
    riscv_asm = riscv_code(riscv_asm_module)

    mo.accordion({
        "RISC-V Assembly": mo.ui.code_editor(riscv_asm, language="python", disabled=True)
    })
    return riscv_asm,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Compiling to Snitch

        xDSL is also capable of targeting Snitch, and making use of its streaming registers and fixed-repetition loop. We use a different lowering flow from the linalg.generic representation to represent a high-level, structured, but Snitch-specific representation of the code:
        """
    )
    return


@app.cell(hide_code=True)
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
    dead_code_elimination,
    loop_hoist_memref,
    lower_affine,
    memref_streamify,
    pipeline_accordion,
    reconcile_unrealized_casts,
):
    convert_linalg_to_snitch = PipelinePass(
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

    snitch_stream_module, snitch_stream_accordion = pipeline_accordion(
        tuple(("", p) for p in convert_linalg_to_snitch.passes), bufferized_module
    )

    snitch_stream_accordion
    return (
        convert_linalg_to_snitch,
        snitch_stream_accordion,
        snitch_stream_module,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("We can then lower this to assembly that includes assembly instructions from the Snitch-extended ISA:")
    return


@app.cell(hide_code=True)
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

    snitch_asm = riscv_code(snitch_asm_module)

    mo.accordion({
        "Snitch Assembly": mo.ui.code_editor(snitch_asm, language="python", disabled=True)
    })
    return snitch_asm, snitch_asm_module


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Interpreting the assembly using xDSL

        One of the useful features of xDSL is its interpreter. Here we've implemented all the necessary functions to interpret the code at a low level, to check that our compilation is correct. Here's the slider modifying the shape variable defined above, we can slide it to see the result of the code compiled with different input shapes, and interpreted at the RISC-V level.
        """
    )
    return


@app.cell
def __(
    Interpreter,
    OpCounter,
    TypedPtr,
    ctx,
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
    return lhs, n, prod, rhs, riscv_interpreter, riscv_op_counter, riscv_res


@app.cell(hide_code=True)
def __(lhs, mo, n, rank, rhs, riscv_res, shape):
    mo.md(
        f"""
        {rank}

        ```
        Shape:  {shape}
        LHS:    {lhs.float64.get_list(n)}
        RHS:    {rhs.float64.get_list(n)}
        Result: {riscv_res.float64.get_list(n)}
        ```
        """
    )
    return


@app.cell
def __(
    Interpreter,
    OpCounter,
    ctx,
    lhs,
    register_implementations,
    rhs,
    snitch_stream_module,
):
    snitch_op_counter = OpCounter()
    snitch_interpreter = Interpreter(snitch_stream_module, listener=snitch_op_counter)

    register_implementations(snitch_interpreter, ctx, include_wgpu=False)

    (snitch_res,) = snitch_interpreter.call_op("main_graph", (lhs, rhs))
    return snitch_interpreter, snitch_op_counter, snitch_res


@app.cell(hide_code=True)
def __(mo, n, snitch_res):
    mo.md(
        f"""
        We can also interpret the Snitch version of the code to compare the results:

        ```
        Snitch result: {snitch_res.float64.get_list(n)}
        ```
        """
    )
    return


@app.cell(hide_code=True)
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
    The interpreter kept track of the number of times an operation was executed, which we can use as a proxy for performance.

    For example, we can see that one version has many more instructions executed overall ({rv_sum} vs {sn_sum}), and that one version uses explicit load and store instructions, while the other uses the streaming equivalents:

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


@app.cell(hide_code=True)
def __():
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
        convert_linalg_to_loops,
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
        convert_linalg_to_loops,
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


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def __(ModuleOp, mo):
    import html as htmllib

    def html(module: ModuleOp) -> mo.Html:
        return f"""\
        <small><code style="white-space: pre-wrap;">{htmllib.escape(str(module))}</code></small>
        """
        # return mo.as_html(str(module))
    return html, htmllib


@app.cell(hide_code=True)
def __():
    from collections import Counter
    return Counter,


@app.cell(hide_code=True)
def __(Counter, ModuleOp, ModulePass, PipelinePass, ctx, html, mo):
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
            html_res = html(res)
            d[header] = mo.vstack((
                text,
                # mo.plain_text(f"Pass: {p.pipeline_pass_spec()}"),
                mo.md(html_res)
            ))
        return (res, mo.accordion(d))
    return pipeline_accordion, spec_str


@app.cell(disabled=True, hide_code=True)
def __(mo, riscv_asm, riscv_code, snitch_asm_module):
    from difflib import Differ, HtmlDiff, unified_diff

    # bla = Differ().compare("hello\nworld".split(), "bla\nworld".split())
    bla = Differ().compare(riscv_code(snitch_asm_module).splitlines(), riscv_asm.splitlines())
    bb = HtmlDiff().make_table("hello\nworld".split(), "bla\nworld".split())
    # bla = Differ().compare(((1, 2), (2)), ((1, 3), (2)))
    c = "\n".join(unified_diff("hello\nworld".split(), "bla\nworld".split()))
    mo.ui.code_editor("\n".join(bla), language="diff")
    return Differ, HtmlDiff, bb, bla, c, unified_diff


if __name__ == "__main__":
    app.run()
