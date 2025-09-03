import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    import marimo as mo
    import urllib

    # Use the locally built xDSL wheel when running in Marimo
    if sys.platform == 'emscripten':

        # Get the current notebook URL, drop the 'blob' URL components that seem to be added,
        # and add the buildnumber that a makethedocs PR build seems to add. This allows to load
        # the wheel both locally and when deployed to makethedocs. 
        def get_url():
            import re
            url = str(mo.notebook_location()).replace("blob:", "")
            print(f"DEBUG: notebook url (full): {url}")

            url_parsed = urllib.parse.urlparse(url)
            scheme = url_parsed.scheme
            netloc = url_parsed.netloc
            print(f"DEBUG: notebook url (parsed): {url_parsed}")

            directory = str(mo.notebook_dir())
            print(f"DEBUG: notebook dir: {directory}")

            url = re.sub('([^/])/([a-f0-9-]+-[a-f0-9-]+-[a-f0-9-]+-[a-f0-9-]+)', '\\1/', url, count=1)
            buildnumber = re.sub('.*--([0-9+]+).*', '\\1', url, count=1)

            url = scheme + "://" + netloc

            if buildnumber != url:
                url = url + "/" + buildnumber + "/"
            elif url == "https://xdsl.readthedocs.io/":
                url = url + "/latest/"

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
def _(mo):
    pass_list = ["constant-fold-interp","canonicalize"]
    return pass_list

@app.cell
def _(mo):
    slider_choices = ["Before"] + ["After " + name for name in pass_list]
    slider = mo.ui.slider(start=0, stop=len(slider_choices) - 1, label="Slider", value=0)
    slider
    return slider

@app.cell
def _(mo, res, pass_list, slider):
    from xdsl.passes import PassPipeline
    from xdsl.transforms import get_all_passes
    from xdsl.context import Context

    module = res


    module_list = [module.clone()]

    def callback(pass1, module, pass2):
        module_list.append(module.clone())

    pipeline = PassPipeline.parse_spec(get_all_passes(), ",".join(pass_list), callback)

    pipeline.apply(Context(), module)
    module_list.append(module.clone())

    res_str_0 = module_str_to_marimo_md(str(module_list[0]))
    res_str_1 = module_str_to_marimo_md(str(module_list[1]))
    res_str_2 = module_str_to_marimo_md(str(module_list[2]))
    mo.md(res_str_0 + "<br>" + res_str_1 + "<br>" + res_str_2)
    return

@app.cell
def _(mo, slider):
    mo.md(f"slidver.value {slider.value}")
    return

if __name__ == "__main__":
    app.run()
