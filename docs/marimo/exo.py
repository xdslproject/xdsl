import marimo

__generated_with = "0.10.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    return mo, xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exo-Style Scheduling in xDSL""")
    return


@app.cell
def _(test):
    def hello():
        for i in range(100):
            for j in range(200):
                test.op(i,j)
        return
    return (hello,)


@app.cell
def _(test):
    def hello_2():
        for io in range(0, 100, 2):
            for jo in range(0, 200, 5):
                for il in range(4):
                    for jl in range(5):
                        i = io + il
                        j = jo + jl
                        test.op(i,j)
        return
    return (hello_2,)


if __name__ == "__main__":
    app.run()
