import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # Uncomment the following two lines to install local version of xDSL.
    # Adjust version string as required
    import micropip
    return (mo, micropip)

@app.cell
def _(mo, micropip):
    await micropip.install("xdsl @ http://127.0.0.1:8000/xdsl-0.0.0-py3-none-any.whl")

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
def _(mo):
    get_state, set_state = mo.state("")
    return (get_state, set_state)

@app.cell
def _(expr_str, mo, printtest, get_state, set_state):
    from xdsl.listlang import ParseError

    try:
        res = printtest(expr_str)
        set_state(res)
    except ParseError:
        res = get_state()

    print(res)
    mo.md(
    f"""
    {res}
    """
    )

    return


if __name__ == "__main__":
    app.run()
