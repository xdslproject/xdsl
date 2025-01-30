import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Assembly-Level Structured Control Flow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Structured Control Flow and the `riscv_scf` Dialect""")
    return


if __name__ == "__main__":
    app.run()
