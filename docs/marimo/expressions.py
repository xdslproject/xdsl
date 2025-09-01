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

    from xdsl.frontend.listlang.main import program_to_mlir, parse_program
    return mo, parse_program, program_to_mlir


@app.cell
def _(mo):
    mo.md(
        r"""
        A small test-page to test mkdocs-marimo.
    
        ## Input
        """
    )
    return


@app.cell
def _(mo):
    expr_str = mo.ui.text_area(value = "let c = 1 + 2;\nc + 2", rows = 10, full_width = True, debounce=False)
    expr_str
    return (expr_str,)


@app.cell
def _(mo):
    mo.md(r"## Output")
    return


@app.cell
def _(mo):
    get_state, set_state = mo.state("")
    return get_state, set_state


@app.cell
def _(expr_str, get_state, mo, parse_program, program_to_mlir, set_state):
    from xdsl.frontend.listlang.main import ParseError
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

    def printtest(code) -> str:
        output = program_to_mlir(code)
        return module_str_to_marimo_md(output)


    try:
        res = printtest(expr_str.value)
        set_state(res)
    except ParseError:
        res = get_state()

    mo.md(f"{res}")

    return module_str_to_marimo_md, to_mlir


@app.cell
def _(mo):
    mo.md(r"# Passes slider")
    return


@app.cell
def _(expr_str, mo, passes_str, to_mlir):
    from xdsl.passes import PassPipeline
    from xdsl.transforms import get_all_passes
    from xdsl.context import Context

    module = to_mlir(expr_str.value)

    module_list = [module.clone()]

    def callback(pass1, module, pass2):
        module_list.append(module.clone())

    pipeline = PassPipeline.parse_spec(get_all_passes(), passes_str.value, callback)
    pass_names = [p.name for p in pipeline.passes]
    slider_choices = ["Before"] + ["After " + name for name in pass_names]

    pipeline.apply(Context(), module)
    module_list.append(module.clone())

    slider = mo.ui.slider(start=0, stop=len(slider_choices) - 1, label="Slider", value=0)
    return module_list, slider, slider_choices


@app.cell
def _(mo):
    passes_str = mo.ui.text(value = "constant-fold-interp,canonicalize", full_width = True)
    return (passes_str,)


@app.cell
def _(
    mo,
    module_list,
    module_str_to_marimo_md,
    passes_str,
    slider,
    slider_choices,
):
    stack = mo.hstack([slider, mo.md(slider_choices[slider.value])])

    mo.vstack([passes_str, stack, mo.md(module_str_to_marimo_md(str(module_list[slider.value])))])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
