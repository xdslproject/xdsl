import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # Use the locally built xDSL version
    import micropip
    import asyncio

    def get_url():
        import re
        url = str(mo.notebook_location())[5:]
        url = re.sub('([^/])/([a-f0-9-]+)', '\\1/', url, count=1)
        buildnumber = re.sub('.*--([0-9+]+).*', '\\1', url, count=1)
        if buildnumber != url:
            print(buildnumber)
            url = url + buildnumber + "/"

        return url

    asyncio.run(micropip.install("xdsl @ " + get_url() + "/xdsl-0.0.0-py3-none-any.whl"))

    from xdsl.printer import Printer

    return (mo,)


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
def _(expr_str, mo):
    mo.md(f"Expr String: {expr_str}")
    return


if __name__ == "__main__":
    app.run()
