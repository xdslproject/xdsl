import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from xdsl.utils import marimo as xmo
    return (xmo,)


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
    expr_str = mo.ui.code_editor(value = "let c = 1 + 2;\nc + 3", language = "rust", debounce=False)
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
def _(expr_str, get_state, mo, set_state, xmo):
    from xdsl.frontend.listlang.main import ParseError, parse_program
    from xdsl.dialects import builtin
    from xdsl.builder import Builder, InsertPoint

    def to_mlir(code: str) -> builtin.ModuleOp:
        module = builtin.ModuleOp([])
        builder = Builder(InsertPoint.at_start(module.body.block))
        parse_program(code, builder)
        return module

    try:
        def convert(code) -> str:
            output = to_mlir(code.value)
            return output

        res = convert(expr_str)
        set_state(res)
    except ParseError:
        res = get_state()

    res_str = xmo.module_md(res).text.replace("\n", "<br>")

    mo.md(f"{res_str}")
    return (res,)


@app.cell
def _(mo):
    mo.md(r"""# Module post-modification""")
    return


@app.cell
def _():
    pass_list = ["constant-fold-interp","canonicalize"]
    return (pass_list,)


@app.cell
def _(mo, pass_list):
    slider_choices = ["IR before a pass was executed"] + ["IR after " + name for name in pass_list]
    slider = mo.ui.slider(start=0, stop=len(slider_choices) - 1, label="Slider", value=0)
    return slider, slider_choices


@app.cell
def _(mo, slider, slider_choices):
    stack = mo.hstack([slider, mo.md(slider_choices[slider.value])])
    stack
    return


@app.cell
def _(mo, pass_list, res, slider, xmo):
    from xdsl.passes import PassPipeline
    from xdsl.transforms import get_all_passes
    from xdsl.dialects import get_all_dialects
    from xdsl.context import Context

    module = res.clone()


    module_list = [module.clone()]

    def callback(pass1, module, pass2):
        module_list.append(module.clone())

    pipeline = PassPipeline.parse_spec(get_all_passes(), ",".join(pass_list), callback)

    ctx = Context()
    for name, func in get_all_dialects().items():
        ctx.register_dialect(name, func)


    pipeline.apply(ctx, module)
    module_list.append(module.clone())

    res_str_0 = xmo.module_md(module_list[slider.value]).text.replace("\n", "<br>")

    mo.md(res_str_0)
    return


if __name__ == "__main__":
    app.run()
