import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    import marimo as mo

    # Use the locally built xDSL wheel when running in Marimo
    if sys.platform == 'emscripten':

        # Get the current notebook URL, drop the 'blob' URL components that seem to be added,
        # and add the buildnumber that a makethedocs PR build seems to add. This allows to load
        # the wheel both locally and when deployed to makethedocs. 
        def get_url():
            import re
            url = str(mo.notebook_location())[5:]
            directory = str(mo.notebook_dir())
            print(f"DEBUG: notebook url (full): {url}")
            print(f"DEBUG: notebook dir: {directory}")
            url = re.sub('([^/])/([a-f0-9-]+-[a-f0-9-]+-[a-f0-9-]+-[a-f0-9-]+)', '\\1/', url, count=1)
            buildnumber = re.sub('.*--([0-9+]+).*', '\\1', url, count=1)
            if buildnumber != url:
                url = url + buildnumber + "/"

            if url == "https://xdsl.readthedocs.io/":
                url = url + "latest/"

            print(f"DEBUG: notebook url (trimmed): {url}")

            return url

        import micropip
        await micropip.install("xdsl @ " + get_url() + "/xdsl-0.0.0-py3-none-any.whl")

    from xdsl.printer import Printer

    from xdsl.frontend.expression.main import program_to_mlir
    return (mo, program_to_mlir)


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
def _(expr_str, mo, program_to_mlir, get_state, set_state):
    from xdsl.frontend.expression.main import ParseError

    try:
        def printtest(code) -> str:
            output = program_to_mlir(code.value)
            output = output.replace("builtin.module {\n", "")
            output = output.replace("\n", "<br>")
            output = output.replace("}", "")

            return output

        res = printtest(expr_str)
        set_state(res)
    except ParseError:
        res = get_state()

    mo.md(f"{res}")

    return


if __name__ == "__main__":
    app.run()
