import marimo

__generated_with = "0.10.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.parser import Parser
    from xdsl.context import Context
    from xdsl.dialects import arith, func, test, scf, builtin
    return Context, Parser, arith, builtin, func, mo, scf, test, xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exo-Style Scheduling in xDSL""")
    return


@app.cell
def _(Context, arith, builtin, func, scf, test):
    ctx = Context()

    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(test.Test)
    return (ctx,)


@app.cell
def _(Parser, ctx, xmo):
    input_str = """
    func.func @hello() {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index

        scf.for %i = %c0 to %c100 step %c1 {
          scf.for %j = %c0 to %c200 step %c1 {
              "test.op"(%i, %j) : (index, index) -> ()
          }
        }
        func.return
    }
    """

    input_module = Parser(ctx, input_str).parse_module()
    xmo.module_html(input_module)
    return input_module, input_str


@app.cell
def _(Parser, ctx, xmo):
    output_str = """
    func.func @hello_2() {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index

        scf.for %io = %c0 to %c100 step %c4 {
          scf.for %jo = %c0 to %c200 step %c5 {
              scf.for %il = %c0 to %c4 step %c1 {
                  scf.for %jl = %c0 to %c5 step %c1 {
                      %i2 = arith.addi %io, %il : index
                      %j2 = arith.addi %jo, %jl : index
                      "test.op"(%i2, %j2) : (index, index) -> ()
                  }
              }
          }
        }
        func.return
    }
    """

    output_module = Parser(ctx, output_str).parse_module()
    xmo.module_html(output_module)
    return output_module, output_str


if __name__ == "__main__":
    app.run()
