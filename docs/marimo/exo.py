import marimo

__generated_with = "0.12.6"
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


if __name__ == "__main__":
    app.run()
