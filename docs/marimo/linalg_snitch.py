import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import xdsl.utils.marimo as xmo
    # Import all the necessary functionality from xDSL for this notebook
    # If you see an error about xdsl not being defined run this cell manually

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
    from xdsl.builder import ImplicitBuilder
    from xdsl.context import Context
    from xdsl.dialects import arith, func, linalg
    from xdsl.dialects.builtin import AffineMap, AffineMapAttr, MemRefType, ModuleOp, f64
    from xdsl.dialects.riscv import riscv_code
    from xdsl.interpreters.utils.ptr import TypedPtr
    from xdsl.ir import Attribute, Block, Region, SSAValue
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
    )
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.lower_snitch import LowerSnitchPass
    from xdsl.transforms.mlir_opt import MLIROptPass
    from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
    from xdsl.transforms.riscv_scf_loop_range_folding import (
        RiscvScfLoopRangeFoldingPass,
    )
    from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation
    return (
        AffineMap,
        AffineMapAttr,
        Block,
        CanonicalizePass,
        Context,
        ConvertRiscvScfToRiscvCfPass,
        ImplicitBuilder,
        MemRefType,
        ModuleOp,
        PipelinePass,
        RISCVRegisterAllocation,
        Region,
        TypedPtr,
        arith,
        arith_add_fastmath,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_linalg_to_loops,
        convert_linalg_to_memref_stream,
        convert_memref_to_riscv,
        convert_riscv_scf_for_to_frep,
        convert_scf_to_riscv_scf,
        f64,
        func,
        get_all_dialects,
        linalg,
        mo,
        reconcile_unrealized_casts,
        riscv_code,
        xmo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Compiling `linalg` to Snitch

    This notebook walks through compiling micro-kernels defined in `linalg` to RISC-V and RISC-V with extensions for [Snitch](https://pulp-platform.github.io/snitch/), a neural network accelerator.

    _Toggle app view with `⌘` + `.` or `ctrl` + `.`_
    """
    )
    return


@app.cell
def _(
    AffineMap,
    AffineMapAttr,
    Block,
    ImplicitBuilder,
    MemRefType,
    ModuleOp,
    Region,
    a_shape,
    arith,
    b_shape,
    c_shape,
    f64,
    func,
    linalg,
    mo,
    xmo,
):
    a_type = MemRefType(f64, a_shape)
    b_type = MemRefType(f64, b_shape)
    c_type = MemRefType(f64, c_shape)

    kernel_op = func.FuncOp("matmul", ((a_type, b_type, c_type), ()))
    with ImplicitBuilder(kernel_op.body) as (a, b, c):
        # Add name hints to make it easier to track how values are lowered
        a.name_hint = "A"
        b.name_hint = "B"
        c.name_hint = "C"
        body = Region(Block(arg_types = (f64, f64, f64)))
        linalg.GenericOp(
            inputs=(a, b),
            outputs=(c,),
            body=body,
            indexing_maps=(
                AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (m, k))),
                AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (k, n))),
                AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (m, n))),
            ),
            iterator_types=(
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.reduction(),
            )
        )
        with ImplicitBuilder(body) as (a_val, b_val, acc_old_val):
            prod_val = arith.MulfOp(a_val, b_val).result
            acc_new_val = arith.AddfOp(acc_old_val, prod_val).result
            linalg.YieldOp(acc_new_val)
            # Add more name hints to make it easier to track how values are lowered
            a_val.name_hint = "a"
            b_val.name_hint = "b"
            acc_old_val.name_hint = "acc_old"
            prod_val.name_hint = "prod"
            acc_new_val.name_hint = "acc_new"
        func.ReturnOp()

    linalg_module = ModuleOp((kernel_op,))

    mo.md(f"""

    Here is matrix multiplication defined in the `linalg` dialect, with the iteration space decoupled from the computation:

    {xmo.module_html(linalg_module)}
    """)
    return (linalg_module,)


@app.cell
def _(mo):
    min_val = 1
    max_val = 10
    m = mo.ui.slider(min_val, max_val, value=2, label="M")
    n = mo.ui.slider(min_val, max_val, value=2, label="N")
    k = mo.ui.slider(min_val, max_val, value=2, label="K")
    return k, m, n


@app.cell(hide_code=True)
def _(k, m, mo, n):
    mo.md(
        f"""
    We can parametrize the shapes of the matrices operated on:

    {m}{m.value}

    {n}{n.value}

    {k}{k.value}
    """
    )
    return


@app.cell
def _(k, m, mo, n):
    a_shape = (m.value, k.value)
    b_shape = (k.value, n.value)
    c_shape = (m.value, n.value)

    mo.md(
        f"""

        ```
        A: {'x'.join(str(dim) for dim in a_shape)}xf64 B: {'x'.join(str(dim) for dim in b_shape)}xf64 C: {'x'.join(str(dim) for dim in c_shape)}xf64
        ```
        """
    )
    return a_shape, b_shape, c_shape


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Compiling to RISC-V""")
    return


@app.cell
def _(Context, get_all_dialects):
    linalg_ctx = Context()

    for dialect_name, dialect_factory in get_all_dialects().items():
        linalg_ctx.register_dialect(dialect_name, dialect_factory)
    return (linalg_ctx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We can take this representation, and lower to RISC-V-specific dialects:""")
    return


@app.cell
def _(
    PipelinePass,
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_linalg_to_loops,
    convert_memref_to_riscv,
    convert_scf_to_riscv_scf,
    linalg_ctx,
    linalg_module,
    reconcile_unrealized_casts,
    xmo,
):
    lower_to_riscv = PipelinePass(
        [
            convert_linalg_to_loops.ConvertLinalgToLoopsPass(),
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemRefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
        ]
    )

    riscv_ctx, riscv_module, riscv_html = xmo.pipeline_html(
        linalg_ctx, linalg_module, tuple(("", p) for p in lower_to_riscv.passes)
    )

    riscv_html
    return riscv_ctx, riscv_module


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    #### Register allocation

    xDSL provides a register allocator for our RISC-V representation, that works on functions with structured control flow:
    """
    )
    return


@app.cell
def _(
    CanonicalizePass,
    PipelinePass,
    RISCVRegisterAllocation,
    riscv_ctx,
    riscv_module,
    xmo,
):
    allocate_registers = PipelinePass(
        [
            RISCVRegisterAllocation(),
            CanonicalizePass(),
        ]
    )

    regalloc_ctx, regalloc_module, regalloc_html = xmo.pipeline_html(
        riscv_ctx, riscv_module, tuple(("", p) for p in allocate_registers.passes),
    )

    regalloc_html
    return regalloc_ctx, regalloc_module


@app.cell
def _(
    CanonicalizePass,
    ConvertRiscvScfToRiscvCfPass,
    PipelinePass,
    regalloc_ctx,
    regalloc_module,
    xmo,
):
    lower_to_asm = PipelinePass(
        [
            ConvertRiscvScfToRiscvCfPass(),
            CanonicalizePass(),
        ]
    )

    riscv_asm_ctx, riscv_asm_module, assembly_html = xmo.pipeline_html(
        regalloc_ctx, regalloc_module, (("", lower_to_asm),),
    )

    assembly_html
    return (riscv_asm_module,)


@app.cell
def _(mo):
    mo.md("""This representation of the program in xDSL corresponds ~1:1 to RISC-V assembly, and we can use a helper function to print that out.""")
    return


@app.cell(hide_code=True)
def _(mo, riscv_asm_module, riscv_code, xmo):
    riscv_asm = riscv_code(riscv_asm_module)

    mo.md(f"""\
    **RISC-V Assembly:**

    {xmo.asm_html(riscv_asm)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Compiling to Snitch

    xDSL is also capable of targeting Snitch, and making use of its streaming registers and fixed-repetition loop. We use a different lowering flow from the linalg.generic representation to represent a high-level, structured, but Snitch-specific representation of the code:
    """
    )
    return


@app.cell
def _(
    PipelinePass,
    arith_add_fastmath,
    convert_linalg_to_memref_stream,
    convert_riscv_scf_for_to_frep,
    linalg_ctx,
    linalg_module,
    xmo,
):
    from xdsl.transforms.test_lower_linalg_to_snitch import LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES, OPTIMISE_MEMREF_STREAM_PASSES

    convert_linalg_to_snitch = PipelinePass(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemRefStreamPass(),
            arith_add_fastmath.AddArithFastMathFlagsPass(),
            *OPTIMISE_MEMREF_STREAM_PASSES,
            *LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
            convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
        ]
    )

    snitch_stream_ctx, snitch_stream_module, snitch_stream_html = xmo.pipeline_html(
        linalg_ctx, linalg_module, tuple(("", p) for p in convert_linalg_to_snitch.passes),
    )

    snitch_stream_html
    return snitch_stream_ctx, snitch_stream_module


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We can then lower this to assembly that includes assembly instructions from the Snitch-extended ISA:""")
    return


@app.cell
def _(snitch_stream_ctx, snitch_stream_module, xmo):
    from xdsl.transforms.test_lower_linalg_to_snitch import LOWER_SNITCH_STREAM_TO_ASM_PASSES

    snitch_asm_ctx, snitch_asm_module, snitch_asm_html = xmo.pipeline_html(
        snitch_stream_ctx, snitch_stream_module, tuple(("", p) for p in LOWER_SNITCH_STREAM_TO_ASM_PASSES)
    )

    snitch_asm_html
    return (snitch_asm_module,)


@app.cell
def _(k, m, mo, n):
    mo.md(
        f"""
    We can see how changing our input sizes affects the assembly produced:

    {m}{m.value}

    {n}{n.value}

    {k}{k.value}
    """
    )
    return


@app.cell
def _(mo, riscv_code, snitch_asm_module, xmo):
    snitch_asm = riscv_code(snitch_asm_module)

    mo.md(f"""\
    **Snitch Assembly:**

    {xmo.asm_html(snitch_asm)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Interpreting the assembly using xDSL

    One of the useful features of xDSL is its interpreter. Here we've implemented all the necessary functions to interpret the code at a low level, to check that our compilation is correct. Here's the slider modifying the shape variable defined above, we can slide it to see the result of the code compiled with different input shapes, and interpreted at the RISC-V level.
    """
    )
    return


@app.cell
def _(TypedPtr, a_shape, b_shape, c_shape, mo, riscv_ctx, riscv_module):
    from math import prod

    from xdsl.interpreter import Interpreter, OpCounter
    from xdsl.interpreters import register_implementations
    from xdsl.interpreters.shaped_array import ShapedArray

    a_len = prod(a_shape)
    b_len = prod(b_shape)
    c_len = prod(c_shape)

    a_shaped = ShapedArray(TypedPtr.new_float64([i + 1 for i in range(a_len)]), a_shape)
    b_shaped = ShapedArray(TypedPtr.new_float64([(i + 1) / 100 for i in range(b_len)]), b_shape)
    riscv_c_shaped = ShapedArray(TypedPtr.new_float64([0.0] * c_len), c_shape)

    riscv_op_counter = OpCounter()
    riscv_interpreter = Interpreter(riscv_module, listeners=(riscv_op_counter,))

    register_implementations(riscv_interpreter, riscv_ctx)

    riscv_interpreter.call_op("matmul", (a_shaped.data_ptr.raw, b_shaped.data_ptr.raw, riscv_c_shaped.data_ptr.raw))

    mo.md(f"""
    **RISC-V Results:**

    A: {a_shaped}

    B: {b_shaped}

    C: {riscv_c_shaped}
    """)
    return (
        Interpreter,
        OpCounter,
        ShapedArray,
        a_shaped,
        b_shaped,
        c_len,
        register_implementations,
        riscv_c_shaped,
        riscv_op_counter,
    )


@app.cell
def _(
    Interpreter,
    OpCounter,
    ShapedArray,
    TypedPtr,
    a_shaped,
    b_shaped,
    c_len,
    c_shape,
    mo,
    register_implementations,
    riscv_c_shaped,
    snitch_stream_ctx,
    snitch_stream_module,
):
    snitch_op_counter = OpCounter()
    snitch_interpreter = Interpreter(
        snitch_stream_module, listeners=(snitch_op_counter,)
    )

    snitch_c_shaped = ShapedArray(TypedPtr.new_float64([0.0] * c_len), c_shape)

    register_implementations(snitch_interpreter, snitch_stream_ctx)

    snitch_interpreter.call_op(
        "matmul",
        (
            a_shaped.data_ptr.raw,
            b_shaped.data_ptr.raw,
            snitch_c_shaped.data_ptr.raw,
        ),
    )

    mo.md(f"""

    **Snitch Results:**

    A: {a_shaped}

    B: {b_shaped}

    C: {riscv_c_shaped}
    """)
    return (snitch_op_counter,)


@app.cell(hide_code=True)
def _(k, m, mo, n, riscv_op_counter, snitch_op_counter):
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

    {m}{m.value}

    {n}{n.value}

    {k}{k.value}

    ```
    {rows}
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
