import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from xdsl.utils import marimo as xmo

    return (xmo,)


@app.cell(hide_code=True)
def _(xmo):
    # Depend on xmo so marimo keeps `return (xmo,)` in the sentinel cell above.
    # Docs export replaces that exact cell body (SYNC_XDSL_IMPORT).
    _ = xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 0: Compiling and Running Toy

    Here is a simple program in the Toy programming language running in a RISC-V emulator,
    compiled using xDSL.
    Try changing the program and observing the output.

    The same compiler is available via the `toyc` CLI; see examples under
    [`docs/Toy/examples/`](https://github.com/xdslproject/xdsl/tree/main/docs/Toy/examples).
    """)
    return


@app.cell
def _():
    from toy.compiler import compile
    from toy.riscv_emulator import emulate_riscv

    from xdsl.utils.exceptions import VerifyException

    program = """
    def main() {
      # Define a variable `a` with shape <2, 3>, initialized with the literal value.
      # The shape is inferred from the supplied literal.
      var a = [[1, 2, 3], [4, 5, 6]];

      # The literal tensor is implicitly reshaped: defining new variables is the way
      # to reshape tensors (element count must match).
      var b<3, 2> = [1, 2, 3, 4, 5, 6];

      # There is a built-in print instruction to display the contents of the tensor
      print(b);

      # Reshapes are implicit on assignment
      var c<2, 3> = b;

      # There are + and * operators for pointwise addition and multiplication
      var d = a + c;

      print(d);
    }
    """

    try:
        code = compile(program)
        res = emulate_riscv(code)
    except VerifyException as e:
        res = str(e)

    print(res)
    return


if __name__ == "__main__":
    app.run()
