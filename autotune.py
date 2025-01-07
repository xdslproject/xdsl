import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import xdsl.utils.marimo as xmo
    return (xmo,)


if __name__ == "__main__":
    app.run()
