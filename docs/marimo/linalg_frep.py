import marimo

__generated_with = "0.4.11"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import re

    from xdsl.dialects.builtin import ModuleOp

    regex = r"^  "

    def pp(module: ModuleOp) -> str:
        """
        Helper function to print modules for paper and slides
        """
        original = str(module)
        no_module = original[:-1].replace("builtin.module {", "")
        dedented = re.sub(regex, "", no_module, 0, re.MULTILINE)
        return (dedented
            .replace("riscv.", "rv.")
            .replace("riscv_func.", "rv_func.")
            .replace("riscv_scf.", "rv_scf."))
    return ModuleOp, pp, re, regex


@app.cell
def __(mo):
    mo.md("""
    # Using xDSL to Lower Matrix Multiplication from `linalg` to Snitch

    This notebook is a detailed guide through the steps involved in lowering a high-level definition of matrix multiplication to Snitch.

    The current state of the lowering is unfinished, as some parts still need to be implemented.
    The source and target, however, are well defined, so we effectively have three parts:

    1. Top-down, going from the linalg definition to the closest abstraction we have to Snitch
    2. fantasy IR that we'd like to be able to reach in the middle
    3. Bottom-up, the highest-level IR that we can write by hand that we know lowers to optimal code for our semantics
    """)
    return


@app.cell
def __(mo):
    linalg_ir = """\
    func.func public @matmul(%X: memref<8x8xf64>,
                             %Y: memref<8x8xf64>,
                             %Z: memref<8x8xf64>) {
        linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
        } ins(%X, %Y : memref<8x8xf64>, memref<8x8xf64>) outs(%Z : memref<8x8xf64>) {
        ^bb0(%x: f64, %y: f64, %z: f64):
            %r0 = arith.mulf %x, %y : f64
            %r1 = arith.addf %z, %r0 : f64
            linalg.yield %r1 : f64
        }
      func.return
    }
    """

    mo.md(f"""
    ## Top-down

    First we define the input IR:
    ``` mlir
    {linalg_ir}
    ```
    """)
    return linalg_ir,


@app.cell
def __(ModuleOp):
    from xdsl.ir import MLContext
    from xdsl.passes import ModulePass

    def apply(p: ModulePass, m: ModuleOp, ctx: MLContext) -> ModuleOp:
        r = m.clone()
        p.apply(ctx, r)
        return r
    return MLContext, ModulePass, apply


@app.cell
def __(MLContext):
    from xdsl.tools.command_line_tool import get_all_dialects

    ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    return ctx, dialect_factory, dialect_name, get_all_dialects


@app.cell
def __(ctx, linalg_ir):
    from xdsl.parser import Parser

    input = Parser(ctx, linalg_ir).parse_module()
    return Parser, input


@app.cell
def __(apply, ctx, input, mo, pp):
    from xdsl.transforms.convert_linalg_to_memref_stream import ConvertLinalgToMemrefStreamPass

    m_g = apply(ConvertLinalgToMemrefStreamPass(), input, ctx)

    mo.md(f"""
    The first step is to convert the linalg generic to memref stream.
    The primary difference is that the iteration bounds are stored on the `memref_stream.generic` op.
    `linalg.generic` constructs the iteration bounds from the sizes of the inputs + the iteration maps, leading to some awkward approaches like empty tensors used just to define the pooling or convolution regions.

    ``` mlir
    {pp(m_g)}
    ```
    """)
    return ConvertLinalgToMemrefStreamPass, m_g


@app.cell
def __(apply, ctx, m_g, mo, pp):
    from xdsl.transforms.memref_streamify import MemrefStreamifyPass

    streamified_m = apply(MemrefStreamifyPass(), m_g, ctx)

    mo.md(f"""
    The next step is to split the access patterns from the computation with the `{MemrefStreamifyPass.name}` pass.

    We get the following IR, with the generic now taking streams as inputs:

    ``` mlir
    {pp(streamified_m)}
    ```
    """)
    return MemrefStreamifyPass, streamified_m


@app.cell
def __(apply, ctx, mo, streamified_m):
    from xdsl.transforms.convert_memref_stream_to_loops import ConvertMemrefStreamToLoopsPass

    loopified_m = apply(ConvertMemrefStreamToLoopsPass(), streamified_m, ctx)

    mo.md(f"""
    Now we can lower to loops. Note that for optimal performance, we want as few memory accesses as possible.
    `linalg.generic` models a perfectly nested loop nest, relying on downstream optimisations to extract redundant accesses out of the inner loop.
    We want to avoid optimisations as much as possible,
    lowering directly to IR that we believe will lower to optimal code.

    Here is the IR that we get lowering to perfectly nested loops:

    ``` mlir
    {str(loopified_m)}
    ```
    """)
    return ConvertMemrefStreamToLoopsPass, loopified_m


@app.cell
def __(mo):
    mo.md("""
    ## Fantasy Land

    The next few steps are not fully implemented
    """)
    return


@app.cell
def __(Parser, ctx, mo, pp):
    imperfect_nest_ir = """
    func.func public @matmul(%0 : memref<8x8xf64>, %1 : memref<8x8xf64>, %2 : memref<8x8xf64>) {
        memref_stream.streaming_region {bounds = [8, 8, 8], indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1, d2)>]} ins(%0, %1 : memref<8x8xf64>, memref<8x8xf64>) {
        ^0(%3 : !stream.readable<f64>, %4 : !stream.readable<f64>):
          %5 = arith.constant 8 : index
          %6 = arith.constant 8 : index
          %7 = arith.constant 8 : index
          %8 = arith.constant 0 : index
          %9 = arith.constant 1 : index
          scf.for %10 = %8 to %5 step %9 {
            scf.for %11 = %8 to %6 step %9 {
              %13 = memref.load %2[%10, %11] : memref<8x8xf64>
              %14 = scf.for %12 = %8 to %7 step %9 iter_args(%15 = %13) -> (f64) {
                %16 = memref_stream.read from %3 : f64
                %17 = memref_stream.read from %4 : f64
                %18 = arith.mulf %16, %17 : f64
                %19 = arith.addf %13, %18 : f64
                scf.yield %19 : f64
              }
              memref.store %14, %2[%10, %11] : memref<8x8xf64>
            }
          }
        }
        func.return
    }
    """

    imperfect_nest_m = Parser(ctx, imperfect_nest_ir).parse_module()

    mo.md(f"""

    **TODO 1**

    But what if we had imperfect nest loop lowering? The code would look something like this, with the output manipulation done outside of the inner loop:

    ``` mlir
    {pp(imperfect_nest_m)}
    ```
    """)
    return imperfect_nest_ir, imperfect_nest_m


@app.cell
def __(apply, ctx, imperfect_nest_m, mo, pp):
    from xdsl.backend.riscv.lowering import (
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
    )
    from xdsl.passes import PipelinePass
    from xdsl.transforms.convert_memref_stream_to_snitch_stream import ConvertMemrefStreamToSnitch
    from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

    to_riscv_pipeline = PipelinePass([
        convert_arith_to_riscv.ConvertArithToRiscvPass(),
        convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
        convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
        convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
        ConvertMemrefStreamToSnitch(),
        ReconcileUnrealizedCastsPass(),
    ])

    rv_loops_m = apply(to_riscv_pipeline, imperfect_nest_m, ctx)

    mo.md(f"""
    We can then lower the code to riscv:

    ``` mlir
    {pp(rv_loops_m)}
    ```
    """)
    return (
        ConvertMemrefStreamToSnitch,
        PipelinePass,
        ReconcileUnrealizedCastsPass,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
        rv_loops_m,
        to_riscv_pipeline,
    )


@app.cell
def __(apply, ctx, mo, pp, rv_loops_m):
    from xdsl.transforms.riscv_cse import RiscvCommonSubexpressionElimination

    rv_loops_cse_m = apply(RiscvCommonSubexpressionElimination(), rv_loops_m, ctx)

    mo.md(f"""
    The glaring inefficiency here is all the pointer offset calculations inside the loop.
    In reality, we are reading from and loading to offsets that are exactly 8 bytes from each other, in order.
    In order to simplify this we need to make a couple of optimisations, starting with common subexpression elimination ({RiscvCommonSubexpressionElimination.name})

    ``` mlir
    {pp(rv_loops_cse_m)}
    ```
    """)
    return RiscvCommonSubexpressionElimination, rv_loops_cse_m


@app.cell
def __(apply, ctx, mo, pp, rv_loops_cse_m):
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass

    rv_canonicalized_m = apply(CanonicalizePass(), rv_loops_cse_m, ctx)
    rv_loops_folded_m = apply(RiscvScfLoopRangeFoldingPass(), rv_canonicalized_m, ctx)
    rv_loops_folded_m = apply(CanonicalizePass(), rv_loops_folded_m, ctx)

    mo.md(f"""
    We can then simplify the loop logic with {RiscvScfLoopRangeFoldingPass.name}, and some canonicalization:

    ``` mlir
    {pp(rv_loops_folded_m)}
    ```
    """)
    return (
        CanonicalizePass,
        RiscvScfLoopRangeFoldingPass,
        rv_canonicalized_m,
        rv_loops_folded_m,
    )


@app.cell
def __(apply, ctx, mo, pp, rv_loops_folded_m):
    from xdsl.transforms.riscv_scf_loop_fusion import RiscvScfLoopFusionPass

    fused_m = apply(RiscvScfLoopFusionPass(), rv_loops_folded_m, ctx)


    mo.md(f"""

    The key pattern to spot there is the nested loop, where the inner loop index iterates until the step of the outer loop.
    These loops can be fused, to give this:

    ``` mlir
    {pp(fused_m)}
    ```
    """)
    return RiscvScfLoopFusionPass, fused_m


@app.cell
def __(
    CanonicalizePass,
    RiscvScfLoopRangeFoldingPass,
    apply,
    ctx,
    fused_m,
    mo,
    pp,
):
    rv_canonicalized_2_m = apply(CanonicalizePass(), fused_m, ctx)
    rv_loops_folded_2_m = apply(RiscvScfLoopRangeFoldingPass(), rv_canonicalized_2_m, ctx)
    rv_loops_folded_2_m = apply(CanonicalizePass(), rv_loops_folded_2_m, ctx)


    mo.md(f"""
    We can then do some loop range folding to eliminate the rest of the unnecessary code in the for loop:

    ``` mlir
    {pp(rv_loops_folded_2_m)}
    ```
    """)
    return rv_canonicalized_2_m, rv_loops_folded_2_m


@app.cell
def __(mo):
    mo.md("""
    ## Bottom-up

    The next steps are fully functional, meaning that we are confident that the assembly generated when lowering from a starting IR yields the assembly we have tested to be optimal on our target platform.
    """)
    return


@app.cell
def __(apply, ctx, mo, pp, rv_loops_folded_2_m):
    from xdsl.transforms.test_lower_snitch_stream_to_asm import TEST_LOWER_LINALG_TO_SNITCH_PASSES

    pass_results = ""
    remaining_m = rv_loops_folded_2_m

    for p_class in TEST_LOWER_LINALG_TO_SNITCH_PASSES:
        p = p_class()
        remaining_m = apply(p, remaining_m, ctx)
        pass_results += f"""
    {p.name}

    ``` mlir
    {pp(remaining_m)}
    ```
    """


    mo.md(f"""
    The rest of the lowering is handled by the `test-lower-linalg-to-snitch` compound pass, here are the individual IRs:

    {pass_results}
    """)
    return (
        TEST_LOWER_LINALG_TO_SNITCH_PASSES,
        p,
        p_class,
        pass_results,
        remaining_m,
    )


if __name__ == "__main__":
    app.run()
