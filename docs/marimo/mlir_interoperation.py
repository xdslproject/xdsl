import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import subprocess
    return mo, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # xDSL-MLIR Interoperation Tutorial

    This tutorial aims to showcase a simple pipeline of actions to unlock MLIR optimisations when lowering from xDSL.
    This tutorial can help users getting familiar with the xDSL-MLIR interoperation. We will start from a higher level of xDSL abstraction, lower to MLIR generic format, apply an optimisation and the return to xDSL-land.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem Setup

    We start by writing a simple example consisting of adding integers.
    We are writing this example using constructs that are supported in xDSL.

    Furthermore, we create 4 integers, namely a, b, c, d.
    Then we just accumulate by the simple following pseudocode and print the result:


    ```bash
    a = 1
    b = 2
    c = a + b
    d = a + b
    e = c + d
    print(e)
    ```
    """
    )
    return


@app.cell
def _():
    from xdsl.dialects.arith import AddiOp, ConstantOp
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32
    from xdsl.dialects.vector import PrintOp
    from xdsl.ir import Block, Region

    # Define two integer constants
    a = ConstantOp(IntegerAttr(1, 32), i32)
    b = ConstantOp(IntegerAttr(2, 32), i32)

    # Operations on these constants
    c = AddiOp(a, b)
    d = AddiOp(a, b)
    e = AddiOp(c, d)
    f = PrintOp(e)

    # Create Block from operations and Region from blocks
    block0 = Block([a, b, c, d, e, f])
    region0 = Region(block0)

    # Create an Operation from the region
    op = ModuleOp(region0)
    return (op,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Using xDSLs printer we can print this operation.
    For convenience we provide a file called `source.mlir` with the code printed below
    """
    )
    return


@app.cell
def _(op):
    from xdsl.printer import Printer

    # Print in xdsl format
    printer = Printer()
    printer.print_op(op)
    return


@app.cell
def _(mo, subprocess):
    # Cross-check file content
    source_file = mo.notebook_dir() / "source.mlir"
    subprocess.run(["cat", source_file])
    return (source_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, let's try to benefit from some MLIR optimization.
    For this example, we will use the [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination).

    See some documentation here: [mlir.llvm CSE docs](https://mlir.llvm.org/docs/Passes/#-cse-eliminate-common-sub-expressions).

    Assuming you have already `mlir-opt` installed in your machine:
    """
    )
    return


@app.cell(hide_code=True)
def _(subprocess):
    mlir_opt_tool = "mlir-opt"
    is_mlir_opt_available = False

    try:
        subprocess.run([mlir_opt_tool, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        is_mlir_opt_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if is_mlir_opt_available:
        print(f"{mlir_opt_tool} is available.")
    else:
        print(f"{mlir_opt_tool} is not available.")
    return (mlir_opt_tool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""you can apply the CSE optimization using the following command:""")
    return


@app.cell
def _(mlir_opt_tool, source_file, subprocess):
    # mlir-opt --cse --mlir-print-op-generic
    ps1 = subprocess.run([mlir_opt_tool, source_file, "--cse", "--mlir-print-op-generic"], capture_output=True)
    print(ps1.stdout.decode('utf-8').strip())
    return (ps1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can clearly see in the optimized output that after CSE we do not need to calculate:

    ```
    arith.addi"(%0, %1) : (i32, i32) -> i32
    ```

    twice!

    Now can we back to xDSL? Yes we can!
    """
    )
    return


@app.cell
def _(ps1, subprocess):
    subprocess.run(["xdsl-opt"], input=ps1.stdout)
    return


if __name__ == "__main__":
    app.run()
