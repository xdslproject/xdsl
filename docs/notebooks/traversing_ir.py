import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from xdsl.ir import Dialect
    from collections import defaultdict
    from xdsl.dialects import builtin, func, arith, scf
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.utils import marimo as xmo
    from collections import Counter
    from xdsl.ir import OpResult
    return (
        Context,
        Counter,
        Dialect,
        OpResult,
        Parser,
        arith,
        builtin,
        defaultdict,
        func,
        mo,
        scf,
        xmo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Traversing IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Parsing IR""")
    return


@app.cell(hide_code=True)
def _():
    sum_of_squares_text = """\
    func.func @sum_of_squares(%n: index) -> index {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %res = scf.for %i = %zero to %n step %one iter_args(%acc_in = %zero) -> (index) {
        %square = arith.muli %i, %i : index
        %acc_out = arith.addi %acc_in, %square : index
        scf.yield %acc_out : index
      }
      func.return %res : index
    }\
    """
    return (sum_of_squares_text,)


@app.cell(hide_code=True)
def _(mo, sum_of_squares_text):
    mo.md(
        fr"""
    In this notebook, we'll be looking at the structure of the following module,
    which implements a sum-of-squares function:

    {mo.ui.code_editor(sum_of_squares_text, language="javascript", disabled=True)}
    """
    )
    return


@app.cell
def _(Context, arith, builtin, func, scf):
    # The context stores the available abstractions
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    return (ctx,)


@app.cell
def _(Parser, ctx, sum_of_squares_text, xmo):
    sum_of_squares_module = Parser(ctx, sum_of_squares_text).parse_module()

    # We can then parse and reprint the same module
    xmo.module_html(sum_of_squares_module)
    return (sum_of_squares_module,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Walking the IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The Operation class exposes a `walk` function, which lets us visit the IR from top to bottom.

    For example, to get the names of all the operations in our module, we can write a snippet like this:
    """
    )
    return


@app.cell
def _(sum_of_squares_module):
    print([op.name for op in sum_of_squares_module.walk()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    These helpers will be useful for the exercises below:

    `Operation`:

    * `results: Sequence[OpResult]`
    * `operands: Sequence[SSAValue]`

    `Block`:

    * `parent_op: Operation | None`
    * `args: Sequence[BlockArgument]`

    `SSAValue`:

    * `uses: set[Use]`
    * `owner: Operation | Block`

    `OpResult` and `BlockArgument` are subclasses of `SSAValue`.

    `Use`:

    * `operation: Operation`
    """
    )
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
def _(mo, sum_of_squares_module, unique_operations):
    try:
        _sorted = "{" + ", ".join(sorted(unique_operations(sum_of_squares_module))) + "}"
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    mo.md(
        fr"""
        ### Exercise 1. Unique Operations

        Modify the function below to return the set of operations used in the module.

        ```
        Expected: {{arith.addi, arith.constant, arith.muli, builtin.module, func.func, func.return, scf.for, scf.yield}}
        Result:   {_res}
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
def _(builtin):
    # Solution

    def _unique_operations(module: builtin.ModuleOp) -> set[str]:
        return {op.name for op in module.walk()}
    return


@app.cell(hide_code=True)
def _(mo, operation_counts, sum_of_squares_module):
    try:
        _res = str(operation_counts(sum_of_squares_module))
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    mo.md(
        fr"""
        ### Exercise 2. Operation Counter

        Modify the function below to return the number of instances of the operations in the module.

        ```
        Expected: {{'builtin.module': 1, 'func.func': 1, 'arith.constant': 2, 'scf.for': 1, 'arith.muli': 1, 'arith.addi': 1, 'scf.yield': 1, 'func.return': 1}}
        Result:   {_res}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    # `Counter` might come in handy

    def operation_counts(module: builtin.ModuleOp) -> dict[str, int]:
        return {}
    return (operation_counts,)


@app.cell(hide_code=True)
def _(Counter, builtin):
    # solution

    def _operation_counts(module: builtin.ModuleOp) -> dict[str, int]:
        return dict(Counter(op.name for op in module.walk()))
    return


@app.cell(hide_code=True)
def _(mo, operations_by_dialect, sum_of_squares_module):
    try:
        _unsorted = operations_by_dialect(sum_of_squares_module)
        _sorted = {
            k: sorted(_unsorted[k])
            for k in sorted(_unsorted)
        }
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    mo.md(
        fr"""
        ### Exercise 3. Operations By Dialect

        Modify the function below to return the operations by dialect in the module

        ```
        Expected: {{'arith': ['addi', 'constant', 'constant', 'muli'], 'builtin': ['module'], 'func': ['func', 'return'], 'scf': ['for', 'yield']}}
        Result:   {_res}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    # `defaultdict` might come in handy

    def operations_by_dialect(module: builtin.ModuleOp) -> dict[str, list[str]]:
        return {}
    return (operations_by_dialect,)


@app.cell(hide_code=True)
def _(Dialect, builtin, defaultdict):
    # solution

    def _operations_by_dialect(module: builtin.ModuleOp) -> dict[str, list[str]]:
        res = defaultdict(list)

        for op in module.walk():
            d, o = Dialect.split_name(op.name)
            res[d].append(o)
        return dict(res)
    return


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
def _(sum_of_squares_module):
    all_operands = set(val.name_hint for op in sum_of_squares_module.walk() for val in op.operands)
    all_results = set(val.name_hint for op in sum_of_squares_module.walk() for val in op.results)
    all_block_args = set(
        val.name_hint
        for op in sum_of_squares_module.walk()
        for region in op.regions
        for block in region.blocks
        for val in block.args
    )
    all_ssa_values = all_operands | all_results | all_block_args
    return (all_ssa_values,)


@app.cell(hide_code=True)
def _(all_ssa_values, mo):
    mo.md(
        fr"""
    Here are all the name hints of SSA values in our module:

    ```
    {{{', '.join(sorted(all_ssa_values))}}}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(definition_by_use, mo, sum_of_squares_module):
    try:
        _unsorted = definition_by_use(sum_of_squares_module)
        _sorted = {
            k: sorted(_unsorted[k])
            for k in sorted(_unsorted)
        }
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    mo.md(
        fr"""
        ### Exercise 4. Definition By Use

        Modify the function below to return the operation that defines the value by the operation that uses it.
        If the operand is a block argument, use the name of the parent operation.

        Note: the `SSAValue` class provides the `owner` helper that might be useful.


        ```
        Expected: {{'arith.addi': ['arith.muli', 'scf.for'], 'arith.muli': ['scf.for', 'scf.for'], 'func.return': ['scf.for'], 'scf.for': ['arith.constant', 'arith.constant', 'arith.constant', 'func.func'], 'scf.yield': ['arith.addi']}}
        Result:   {_res}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    def definition_by_use(module: builtin.ModuleOp) -> dict[str, list[str]]:
        return {}
    return (definition_by_use,)


@app.cell(hide_code=True)
def _(OpResult, builtin, defaultdict):
    # Solution

    def _definition_by_use(module: builtin.ModuleOp) -> dict[str, list[str]]:
        res = defaultdict(list)

        for use_op in module.walk():
            for operand in use_op.operands:
                if isinstance(operand, OpResult):
                    op = operand.op
                else:
                    op = operand.block.parent_op()

                res[use_op.name].append(op.name)
        return dict(res)
    return


@app.cell(hide_code=True)
def _(mo, sum_of_squares_module, uses_by_definition):
    try:
        _unsorted = uses_by_definition(sum_of_squares_module)
        _sorted = {
            k: sorted(_unsorted[k])
            for k in sorted(_unsorted)
        }
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    mo.md(
        fr"""
        ### Exercise 5. Uses By Definition

        Modify the function below to return the operations that use the result by the operation that defines it.

        Note: the `SSAValue` class provides a `uses` helper that might be useful.

        ```
        Expected: {{'arith.addi': ['scf.yield'], 'arith.constant': ['scf.for', 'scf.for', 'scf.for'], 'arith.muli': ['arith.addi'], 'scf.for': ['func.return']}}
        Result:   {_sorted}
        ```
        """
    )
    return


@app.cell
def _(builtin):
    def uses_by_definition(module: builtin.ModuleOp) -> dict[str, list[str]]:
        return {}
    return (uses_by_definition,)


@app.cell(hide_code=True)
def _(builtin, defaultdict):
    # Solution

    def _uses_by_definition(module: builtin.ModuleOp) -> dict[str, list[str]]:
        res = defaultdict(list)

        for def_op in module.walk():
            for result in def_op.results:
                for use in result.uses:
                    res[def_op.name].append(use.operation.name)

        return dict(res)
    return


if __name__ == "__main__":
    app.run()
