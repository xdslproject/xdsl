import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from xdsl.utils import marimo as xmo
    return (xmo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Pattern Description Language (PDL)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    One of the most frequent kinds of transformations on intermediate representations in compilers are local, "peephole", rewrites, which transform an operation by inspecting its local context only.
    An example of this might be rewriting `arith.add` into `llvm.add`, which can be done just by inspecting the type of the operation, or local optimisations such as rewriting an `arith.add` with `0` to the other operand.
    In both MLIR and xDSL these can be written using the native language API (C++ or Python), a flexible and productive approach with two important flaws: reasoning about these rewrites requires reasoning about the semantics of the host language, which is famously difficult for both C++ and Python, and generating them is much less convenient than generating MLIR IR directly.
    The `pdl` dialect addresses both of these issues.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PDL Patterns""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is an example pattern in PDL to rewrite `x + x` to `x * 2`:""")
    return


@app.cell
def _():
    return


app._unparsable_cell(
    r"""

    func.func @impl() -> i32 {
      %0 = arith.constant 4 : i32
      %1 = arith.constant 0 : i32
      %2 = arith.addi %0, %1 : i32
      func.return %2 : i32
    }

    pdl.pattern : benefit(2) {
      %0 = pdl.type
      %1 = pdl.operand
      %2 = pdl.attribute = 0 : i32
      %3 = pdl.operation \"arith.constant\" {\"value\" = %2} -> (%0 : !pdl.type)
      %4 = pdl.result 0 of %3
      %5 = pdl.operation \"arith.addi\" (%1, %4 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
      pdl.rewrite %5 {
        pdl.replace %5 with (%1 : !pdl.value)
      }
    }

    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Input Function

    In this notebook, you'll write patterns in PDL to transform this input function:
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from xdsl.dialects import arith, builtin
    from xdsl.frontend.pyast.context import PyASTContext
    from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

    # Set up the AST parsing context
    ctx = PyASTContext()
    ctx.register_type(float, builtin.f64)
    ctx.register_function(float.__add__, arith.AddfOp)
    ctx.register_function(float.__sub__, arith.SubfOp)
    ctx.register_function(float.__mul__, arith.MulfOp)
    ctx.register_function(float.__truediv__, arith.DivfOp)
    return (ctx,)


@app.cell
def _(ctx):
    @ctx.parse_program
    def main(a: float, b: float, c: float) -> float:
        return (c + (a - a)) / (b / b)
    return (main,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As you can see, it just returns `c`, we'll want to rewrite our program to return `c` directly, without the useless computations.""")
    return


@app.cell
def _(main):
    print(main(1, 2, 3))
    print(main(4, 5, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is the corresponding IR:""")
    return


@app.cell
def _(main, xmo):
    xmo.module_html(main.module)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
