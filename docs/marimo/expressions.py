import marimo

__generated_with = "0.15.0"
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
    expr_str = mo.ui.text_area(value = "let c = 1 + 2;\nc + 3", rows = 10, full_width = True, debounce=False)
    expr_str
    return (expr_str,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Output""")
    return


@app.cell
def _(mo):
    get_state, set_state = mo.state("")
    return get_state, set_state


@app.cell
def _(expr_str, get_state, mo, set_state):
    from xdsl.frontend.listlang.main import ParseError, parse_program
    from xdsl.dialects import builtin
    from xdsl.builder import Builder, InsertPoint

    def to_mlir(code: str) -> builtin.ModuleOp:
        module = builtin.ModuleOp([])
        builder = Builder(InsertPoint.at_start(module.body.block))
        parse_program(code, builder)
        return module

    def module_str_to_marimo_md(module: str) -> str:
        output = module[:].replace("builtin.module {\n", "")
        output = output.replace("\n", "<br>")
        output = output.replace("}", "")
        return output

    try:
        def convert(code) -> str:
            output = to_mlir(code.value)
            return output

        res = convert(expr_str)
        set_state(res)
    except ParseError:
        res = get_state()

    res_str = module_str_to_marimo_md(str(res))

    mo.md(f"{res_str}")

    return res


@app.cell
def _(mo):
    mo.md(r"""# Module post-modification""")
    return


@app.cell
def _(mo, res):
    module = res

    res_str2 = module_str_to_marimo_md(str(module))

    mo.md(res_str2)

if __name__ == "__main__":
    app.run()
