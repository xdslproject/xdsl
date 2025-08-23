import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # Uncomment the following two lines to install local version of xDSL.
    # Adjust version string as required
    import micropip
    await micropip.install("xdsl @ http://xdsl--5103.org.readthedocs.build/5103/xdsl-0.0.0-py3-none-any.whl")

    from xdsl.listlang import printtest
    return (mo, printtest)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    A small test-page to test mkdocs-marimo.

    ## Input
    """
    )
    return


@app.cell
def _(mo):
    expr_str = mo.ui.text(value = "3", debounce=False)
    expr_str
    return (expr_str,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Output""")
    return


@app.cell
def _(expr_str, mo, printtest):
    res = printtest(expr_str)
    print(res)
    mo.md(
    f"""
    {res}
    """
    )

    return


if __name__ == "__main__":
    app.run()
