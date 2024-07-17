import marimo

__generated_with = "0.7.5"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """\
    # Compiling `linalg` to Snitch

    This notebook walks through compiling micro-kernels defined in `linalg` to RISC-V and RISC-V with extensions for [Snitch](https://pulp-platform.github.io/snitch/), a neural network accelerator.
    """
    )
    return


@app.cell
def __():
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import func, linalg
    from xdsl.dialects.builtin import MemRefType, ModuleOp, f64

    a_type = MemRefType(f64, (2, 3))
    b_type = MemRefType(f64, (3, 4))
    c_type = MemRefType(f64, (2, 4))

    kernel_op = func.FuncOp("kernel", ((a_type, b_type, c_type), ()))

    with ImplicitBuilder(kernel_op.body) as (a, b, c):
        linalg.MatmulOp(inputs=(a, b), outputs=(c,))
        func.Return()

    linalg_module = ModuleOp((kernel_op,))

    str(linalg_module)
    return (
        ImplicitBuilder,
        MemRefType,
        ModuleOp,
        a,
        a_type,
        b,
        b_type,
        c,
        c_type,
        f64,
        func,
        kernel_op,
        linalg,
        linalg_module,
    )


if __name__ == "__main__":
    app.run()
