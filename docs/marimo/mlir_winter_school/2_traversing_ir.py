import marimo

__generated_with = "0.10.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 2. Traversing IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Parsing IR""")
    return


@app.cell(hide_code=True)
def _():
    triangle_text = """\
    func.func @triangle(%n: index) -> index {
      %zero = arith.constant 0 : index
      %step = arith.constant 1 : index
      %init = arith.constant 0 : index
      %res = scf.for %i = %zero to %n step %step iter_args(%acc_in = %init) -> (index) {
        %square = arith.muli %i, %i : index
        %acc_out = arith.addi %acc_in, %square : index
        scf.yield %acc_out : index
      }
      func.return %res : index
    }\
    """
    return (triangle_text,)


@app.cell(hide_code=True)
def _(mo, triangle_text):
    mo.md(fr"""
    In this notebook, we'll be looking at the structure of the following module:

    {mo.ui.code_editor(triangle_text, language="javascript", disabled=True)}
    """
    )
    return


@app.cell
def _():
    # Our module contains operations from the following dialects
    from xdsl.dialects import builtin, func, arith, scf
    return arith, builtin, func, scf


@app.cell
def _(arith, builtin, func, scf):
    # The context stores the available abstractions
    from xdsl.context import MLContext

    ctx = MLContext()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    return MLContext, ctx


@app.cell
def _(ctx, triangle_text):
    from xdsl.parser import Parser
    from xdsl.utils import marimo as xmo

    triangle_module = Parser(ctx, triangle_text).parse_module()

    # We can then parse and reprint the same module
    xmo.module_html(triangle_module)
    return Parser, triangle_module, xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Walking the IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Operation class exposes a `walk` function, which lets us iterate over the IR from top to bottom.

        For example, to get the names of all the operations in our module, we can write a snippet like this:
        """
    )
    return


@app.cell
def _(mo, triangle_module):
    mo.md(str([op.name for op in triangle_module.walk()]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exercises""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The rest of the notebook is a series of exercises for you to get acquainted with the API of xDSL.""")
    return


@app.cell(hide_code=True)
def _(mo, triangle_module, unique_operations):
    mo.md(
        fr"""
        ### Exercise 1. Unique Operations

        Modify the function below to return the set of operations used in the module.

        ```
        Expected: {{arith.addi, arith.constant, arith.muli, builtin.module, func.func, func.return, scf.for, scf.yield}}
        Result:   {{{', '.join(sorted(unique_operations(triangle_module)))}}}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    def unique_operations(module: builtin.ModuleOp) -> set[str]:
        return set()
    return (unique_operations,)


@app.cell(hide_code=True)
def _(mo, operation_counts, triangle_module):
    mo.md(
        fr"""
        ### Exercise 2. Operation Counter

        Modify the function below to return the number of instances of the operations in the module.

        ```
        Expected: {{'builtin.module': 1, 'func.func': 1, 'arith.constant': 3, 'scf.for': 1, 'arith.muli': 1, 'arith.addi': 1, 'scf.yield': 1, 'func.return': 1}}
        Result:   {operation_counts(triangle_module)}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    # This might come in handy
    from collections import Counter

    def operation_counts(module: builtin.ModuleOp) -> dict[str, int]:
        return {}
    return Counter, operation_counts


@app.cell(hide_code=True)
def _(mo, operations_by_dialect, triangle_module):
    mo.md(
        fr"""
        ### Exercise 3. Operations By Dialect

        Modify the function below to return the operations by dialect in the module

        ```
        Expected: {{'builtin': ['module'], 'func': ['func', 'return'], 'arith': ['constant', 'constant', 'constant', 'muli', 'addi'], 'scf': ['for', 'yield']}}
        Result:   {operations_by_dialect(triangle_module)}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    # These might come in handy
    from xdsl.ir import Dialect
    from collections import defaultdict

    def operations_by_dialect(module: builtin.ModuleOp) -> dict[str, int]:
        return {}
    return Dialect, defaultdict, operations_by_dialect


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## SSA Values""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        SSA values in IR are either defined as results or block arguments.
        In xDSL, the values have optional name hints, and are numbered in ascending order when printing if the name hint is missing.
        """
    )
    return


@app.cell(hide_code=True)
def _(triangle_module):
    all_operands = set(val.name_hint for op in triangle_module.walk() for val in op.operands)
    all_results = set(val.name_hint for op in triangle_module.walk() for val in op.results)
    all_block_args = set(
        val.name_hint
        for op in triangle_module.walk()
        for region in op.regions
        for block in region.blocks
        for val in block.args
    )
    all_ssa_values = all_operands | all_results | all_block_args
    return all_block_args, all_operands, all_results, all_ssa_values


@app.cell(hide_code=True)
def _(all_ssa_values, mo):
    mo.md(fr"""
    Here are all the name hints of SSA values in our module:

    ```
    {{{', '.join(sorted(all_ssa_values))}}}
    ```
    """
    )
    return


@app.cell
def _(mo, operations_by_dialect, triangle_module):
    mo.md(
        fr"""
        ### Exercise 4. Definition By Use

        Modify the function below to return the operations by dialect in the module

        ```
        Expected: {{'builtin': ['module'], 'func': ['func', 'return'], 'arith': ['constant', 'constant', 'constant', 'muli', 'addi'], 'scf': ['for', 'yield']}}
        Result:   {operations_by_dialect(triangle_module)}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
