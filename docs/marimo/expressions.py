import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
async def _():
    import marimo as mo

    # Use the locally built xDSL wheel when running in Marimo
    if mo.running_in_notebook():

        # Get the current notebook URL, drop the 'blob' URL components that seem to be added,
        # and add the buildnumber that a makethedocs PR build seems to add. This allows to load
        # the wheel both locally and when deployed to makethedocs. 
        def get_url():
            import re
            url = str(mo.notebook_location())[5:]
            url = re.sub('([^/])/([a-f0-9-]+)', '\\1/', url, count=1)
            buildnumber = re.sub('.*--([0-9+]+).*', '\\1', url, count=1)
            if buildnumber != url:
                url = url + buildnumber + "/"

            return url

        import micropip
        await micropip.install("xdsl @ " + get_url() + "/xdsl-0.0.0-py3-none-any.whl")

    from xdsl.printer import Printer

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
    expr_str = mo.ui.text(value = "1 + 2", debounce=False)
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

    mo.md(f"{res}")

    return


if __name__ == "__main__":
    app.run()
