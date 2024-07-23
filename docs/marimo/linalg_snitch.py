import marimo

__generated_with = "0.7.5"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        """
        \
        # Compiling `linalg` to Snitch

        This notebook walks through compiling micro-kernels defined in `linalg` to RISC-V and RISC-V with extensions for [Snitch](https://pulp-platform.github.io/snitch/), a neural network accelerator.

        _Toggle app view with `⌘` + `.` or `ctrl` + `.`_
        """
    )
    return


@app.cell
def __():
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
    from xdsl.context import MLContext
    from xdsl.dialects import arith, func, linalg
    from xdsl.dialects.builtin import AffineMap, AffineMapAttr, MemRefType, ModuleOp, f64
    from xdsl.dialects.riscv import riscv_code
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
        test_lower_snitch_stream_to_asm,
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
        Attribute,
        Block,
        CanonicalizePass,
        ConvertRiscvScfToRiscvCfPass,
        ConvertSnitchStreamToSnitch,
        ImplicitBuilder,
        LowerSnitchPass,
        MLContext,
        MLIROptPass,
        MemRefType,
        ModuleOp,
        PipelinePass,
        RISCVRegisterAllocation,
        Region,
        RiscvScfLoopRangeFoldingPass,
        SSAValue,
        SnitchRegisterAllocation,
        arith,
        arith_add_fastmath,
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
        f64,
        func,
        get_all_dialects,
        linalg,
        loop_hoist_memref,
        lower_affine,
        memref_streamify,
        reconcile_unrealized_casts,
        riscv_code,
        test_lower_snitch_stream_to_asm,
    )


@app.cell
def __(
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
    html,
    linalg,
    mo,
):
    a_type = MemRefType(f64, a_shape)
    b_type = MemRefType(f64, b_shape)
    c_type = MemRefType(f64, c_shape)

    kernel_op = func.FuncOp("matmul", ((a_type, b_type, c_type), ()))
    with ImplicitBuilder(kernel_op.body) as (a, b, c):
        # Add name hints to make it easier to track how values are lowered
        a.name_hint = "A"
        b.name_hint = "B"
        c.name_hing = "C"
        body = Region(Block(arg_types = (f64, f64, f64)))
        linalg.Generic(
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
            prod_val = arith.Mulf(a_val, b_val).result
            acc_new_val = arith.Addf(acc_old_val, prod_val).result
            linalg.YieldOp(acc_new_val)
            # Add more name hints to make it easier to track how values are lowered
            a_val.name_hint = "a"
            b_val.name_hint = "b"
            acc_old_val.name_hint = "acc_old"
            prod_val.name_hint = "prod"
            acc_new_val.name_hint = "acc_new"
        func.Return()

    linalg_module = ModuleOp((kernel_op,))

    mo.md(f"""

    Here is matrix multiplication defined in the `linalg` dialect, with the iteration space decoupled from the computation:

    {html(linalg_module)}
    """)
    return (
        a,
        a_type,
        a_val,
        acc_new_val,
        acc_old_val,
        b,
        b_type,
        b_val,
        body,
        c,
        c_type,
        kernel_op,
        linalg_module,
        prod_val,
    )


@app.cell
def __(mo):
    min_val = 1
    max_val = 10
    m = mo.ui.slider(min_val, max_val, value=2, label="M")
    n = mo.ui.slider(min_val, max_val, value=2, label="N")
    k = mo.ui.slider(min_val, max_val, value=2, label="K")
    return k, m, max_val, min_val, n


@app.cell
def __(k, m, mo, n):
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
def __(k, m, mo, n):
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


@app.cell
def __(mo):
    mo.md("### Compiling to RISC-V")
    return


@app.cell
def __(MLContext, get_all_dialects):
    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx, dialect_factory, dialect_name


@app.cell
def __(mo):
    mo.md("We can take this representation, and lower to RISC-V-specific dialects:")
    return


@app.cell
def __(
    PipelinePass,
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_linalg_to_loops,
    convert_memref_to_riscv,
    convert_scf_to_riscv_scf,
    linalg_module,
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
        tuple(("", p) for p in lower_to_riscv.passes), linalg_module
    )

    riscv_accordion
    return lower_to_riscv, riscv_accordion, riscv_module


@app.cell
def __(mo):
    mo.md(
        """
        #### Register allocation

        xDSL provides a register allocator for our RISC-V representation, that works on functions with structured control flow:
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


@app.cell
def __(mo):
    mo.md("This representation of the program in xDSL corresponds ~1:1 to RISC-V assembly, and we can use a helper function to print that out.")
    return


@app.cell
def __(mo, riscv_asm_module, riscv_code):
    riscv_asm = riscv_code(riscv_asm_module)

    mo.accordion(
        {
            "RISC-V Assembly": mo.ui.code_editor(
                riscv_asm, language="python", disabled=True
            )
        }
    )
    return riscv_asm,


@app.cell
def __(mo):
    mo.md(
        """
        ### Compiling to Snitch

        xDSL is also capable of targeting Snitch, and making use of its streaming registers and fixed-repetition loop. We use a different lowering flow from the linalg.generic representation to represent a high-level, structured, but Snitch-specific representation of the code:
        """
    )
    return


@app.cell
def __(
    PipelinePass,
    arith_add_fastmath,
    convert_linalg_to_memref_stream,
    convert_riscv_scf_for_to_frep,
    linalg_module,
    pipeline_accordion,
):
    from xdsl.transforms.test_lower_memref_stream_to_snitch_stream import (
        TEST_LOWER_MEMREF_STREAM_TO_SNITCH_STREAM,
    )
    from xdsl.transforms.test_optimise_memref_stream import TEST_OPTIMISE_MEMREF_STREAM

    convert_linalg_to_snitch = PipelinePass(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemrefStreamPass(),
            arith_add_fastmath.AddArithFastMathFlagsPass(),
            *TEST_OPTIMISE_MEMREF_STREAM,
            *TEST_LOWER_MEMREF_STREAM_TO_SNITCH_STREAM,
            convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
        ]
    )

    snitch_stream_module, snitch_stream_accordion = pipeline_accordion(
        tuple(("", p) for p in convert_linalg_to_snitch.passes), linalg_module
    )

    snitch_stream_accordion
    return (
        TEST_LOWER_MEMREF_STREAM_TO_SNITCH_STREAM,
        TEST_OPTIMISE_MEMREF_STREAM,
        convert_linalg_to_snitch,
        snitch_stream_accordion,
        snitch_stream_module,
    )


@app.cell
def __(mo):
    mo.md("We can then lower this to assembly that includes assembly instructions from the Snitch-extended ISA:")
    return


@app.cell
def __(
    pipeline_accordion,
    snitch_stream_module,
    test_lower_snitch_stream_to_asm,
):
    snitch_asm_module, snitch_asm_accordion = pipeline_accordion(
        tuple(("", p) for p in test_lower_snitch_stream_to_asm.TEST_LOWER_SNITCH_STREAM_TO_ASM_PASSES), snitch_stream_module
    )

    snitch_asm_accordion
    return snitch_asm_accordion, snitch_asm_module


@app.cell
def __(k, m, mo, n):
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
def __(mo, riscv_code, snitch_asm_module):
    snitch_asm = riscv_code(snitch_asm_module)

    mo.accordion(
        {
            "Snitch Assembly": mo.ui.code_editor(
                snitch_asm, language="python", disabled=True
            )
        }
    )
    return snitch_asm,


@app.cell
def __(ModuleOp, mo):
    import html as htmllib

    def html(module: ModuleOp) -> mo.Html:
        return f"""\
        <small><code style="white-space: pre-wrap;">{htmllib.escape(str(module))}</code></small>
        """
        # return mo.as_html(str(module))
    return html, htmllib


@app.cell
def __():
    from collections import Counter
    return Counter,


@app.cell
def __(Counter, ModuleOp, ModulePass, PipelinePass, ctx, html, mo):
    def spec_str(p: ModulePass) -> str:
        if isinstance(p, PipelinePass):
            return ",".join(str(c.pipeline_pass_spec()) for c in p.passes)
        else:
            return str(p.pipeline_pass_spec())

    def pipeline_accordion(
        passes: tuple[tuple[mo.Html, ModulePass], ...], module: ModuleOp
    ) -> tuple[ModuleOp, mo.Html]:
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
            d[header] = mo.vstack(
                (
                    text,
                    mo.md(html_res),
                )
            )
        return (res, mo.accordion(d))
    return pipeline_accordion, spec_str


if __name__ == "__main__":
    app.run()
