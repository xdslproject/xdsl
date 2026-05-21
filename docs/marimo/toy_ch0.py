import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from xdsl.utils import marimo as xmo

    return (xmo,)


@app.cell(hide_code=True)
def _(xmo):
    # There has to be a user of xmo for the website build
    _ = xmo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 0: Compiling and Running Toy

    Here is a simple program in the Toy programming language running in a RISC-V emulator,
    compiled using xDSL.
    Try changing the program and observing the output:
    """)
    return


@app.cell
def _(mo):
    program_area = mo.ui.text_area("""\
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
    """)

    program_area
    return (program_area,)


@app.cell
def _(mo, program_area):
    from toy.compiler import compile
    from toy.riscv_emulator import emulate_riscv
    from xdsl.utils.exceptions import VerifyException

    from contextlib import redirect_stdout
    import io

    f = io.StringIO()
    with redirect_stdout(f):
        try:
            _code = compile(program_area.value)
            emulate_riscv(_code)
        except VerifyException as _e:
            print(_e)
    s = f.getvalue()

    mo.ui.md(f)
    return


if __name__ == "__main__":
    app.run()
