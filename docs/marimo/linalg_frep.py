import marimo

__generated_with = "0.2.5"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


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
def __():
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import MLContext
    from xdsl.passes import ModulePass

    def apply(p: ModulePass, m: ModuleOp, ctx: MLContext) -> ModuleOp:
        r = m.clone()
        p.apply(ctx, r)
        return r
    return MLContext, ModuleOp, ModulePass, apply


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
def __(apply, ctx, input, mo):
    from xdsl.transforms.convert_linalg_to_memref_stream import ConvertLinalgToMemrefStreamPass

    m_g = apply(ConvertLinalgToMemrefStreamPass(), input, ctx)

    mo.md(f"""
    The first step is to convert the linalg generic to memref stream.
    The primary difference is that the iteration bounds are stored on the `memref_stream.generic` op.
    `linalg.generic` constructs the iteration bounds from the sizes of the inputs + the iteration maps, leading to some awkward approaches like empty tensors used just to define the pooling or convolution regions.

    ``` mlir
    {str(m_g)}
    ```
    """)
    return ConvertLinalgToMemrefStreamPass, m_g


@app.cell
def __(apply, ctx, m_g, mo):
    from xdsl.transforms.memref_streamify import MemrefStreamifyPass

    streamified_m = apply(MemrefStreamifyPass(), m_g, ctx)

    mo.md(f"""
    The next step is to split the access patterns from the computation with the `{MemrefStreamifyPass.name}` pass.

    We get the following IR, with the generic now taking streams as inputs:

    ``` mlir
    {str(streamified_m)}
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
def __(Parser, ctx, mo):
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
                %19 = arith.addf %13, %16 : f64
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
    But what if we had imperfect nest loop lowering? The code would look something like this, with the output manipulation done outside of the inner loop:

    ``` mlir
    {str(imperfect_nest_m)}
    ```
    """)
    return imperfect_nest_ir, imperfect_nest_m


@app.cell
def __(ReconcileUnrealizedCastsPass, apply, ctx, imperfect_nest_m, mo):
    from xdsl.backend.riscv.lowering import (
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
    )
    from xdsl.passes import PipelinePass
    from xdsl.transforms.convert_memref_stream_to_snitch_stream import ConvertMemrefStreamToSnitch

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
    {str(rv_loops_m)}
    ```
    """)
    return (
        ConvertMemrefStreamToSnitch,
        PipelinePass,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
        rv_loops_m,
        to_riscv_pipeline,
    )


@app.cell
def __(apply, ctx, mo, rv_loops_m):
    from xdsl.transforms.riscv_cse import RiscvCommonSubexpressionElimination

    rv_loops_cse_m = apply(RiscvCommonSubexpressionElimination(), rv_loops_m, ctx)

    mo.md(f"""
    The glaring inefficiency here is all the pointer offset calculations inside the loop.
    In reality, we are reading from and loading to offsets that are exactly 8 bytes from each other, in order.
    In order to simplify this we need to make a couple of optimisations, starting with common subexpression elimination ({RiscvCommonSubexpressionElimination.name})

    ``` mlir
    {str(rv_loops_cse_m)}
    ```
    """)
    return RiscvCommonSubexpressionElimination, rv_loops_cse_m


@app.cell
def __(mo):
    mo.md("""
    ## Bottom-up

    The next steps are fully functional, meaning that we are confident that the assembly generated when lowering from a starting IR yields the assembly we have tested to be optimal on our target platform.
    """)
    return


@app.cell
def __(Parser, ctx, mo):
    b_u_ir = """\
      func.func @matmul(
        %X : memref<8x8xf64>,
        %Y : memref<8x8xf64>,
        %G : memref<8x8xf64>
      ) {
        %X_moved = builtin.unrealized_conversion_cast %X : memref<8x8xf64> to !riscv.reg<>
        %Y_moved = builtin.unrealized_conversion_cast %Y : memref<8x8xf64> to !riscv.reg<>
        %G_moved = builtin.unrealized_conversion_cast %G : memref<8x8xf64> to !riscv.reg<>


        %c0 = riscv.li 0 : () -> !riscv.reg<>
        %c1 = riscv.li 1 : () -> !riscv.reg<>
        %c8 = riscv.li 8 : () -> !riscv.reg<>
        %c512 = riscv.li 512 : () -> !riscv.reg<>

        "snitch_stream.streaming_region"(%X_moved, %Y_moved) <{
          "stride_patterns" = [
            #snitch_stream.stride_pattern<ub = [8, 8, 8], strides = [8, 0, 64]>,
            #snitch_stream.stride_pattern<ub = [8, 8, 8], strides = [64, 8, 0]>
          ],
          "operandSegmentSizes" = array<i32: 2, 0>
        }> ({
        ^bb0(%X_stream : !stream.readable<!riscv.freg<>>, %Y_stream : !stream.readable<!riscv.freg<>>):
          riscv_scf.for %g_i : !riscv.reg<> = %c0 to %c512 step %c8 {
            %G_dest = riscv.add %G_moved, %g_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            %init = riscv.fld %G_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

            %g = riscv_scf.for %i : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
              %x = riscv_snitch.read from %X_stream : !riscv.freg<>
              %y = riscv_snitch.read from %Y_stream : !riscv.freg<>
              %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
              riscv_scf.yield %res : !riscv.freg<>
            }

            riscv.fsd %G_dest, %g, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

            riscv_scf.yield
          }
        }) : (!riscv.reg<>, !riscv.reg<>) -> ()

        func.return
      }
    """

    b_u_m = Parser(ctx, b_u_ir).parse_module()

    mo.md(f"""
    Here is our partially lowered IR:

    ```
    {str(b_u_m)}
    ```
    """)
    return b_u_ir, b_u_m


@app.cell
def __(apply, b_u_m, ctx, mo):
    from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import ConvertFuncToRiscvFuncPass

    riscv_func_m = apply(ConvertFuncToRiscvFuncPass(), b_u_m, ctx)

    mo.md(f"""

    convert-func-to-riscv-func

    ```
    {str(riscv_func_m)}
    ```
    """)
    return ConvertFuncToRiscvFuncPass, riscv_func_m


@app.cell
def __(apply, ctx, mo, riscv_func_m):
    from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

    riscv_m = apply(ReconcileUnrealizedCastsPass(), riscv_func_m, ctx)

    mo.md(f"""

    With our lowering to riscv complete, we can remove the casts with `reconcile-unrealized-casts

    ```
    {str(riscv_m)}
    ```
    """)
    return ReconcileUnrealizedCastsPass, riscv_m


@app.cell
def __(apply, ctx, mo, riscv_m):
    from xdsl.transforms.test_lower_linalg_to_snitch import TEST_LOWER_LINALG_TO_SNITCH_PASSES

    pass_results = ""
    remaining_m = riscv_m

    for p_class in TEST_LOWER_LINALG_TO_SNITCH_PASSES:
        p = p_class()
        remaining_m = apply(p, remaining_m, ctx)
        pass_results += f"""
    {p.name}

    ``` mlir
    {str(remaining_m)}
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
