import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from xdsl.builder import Builder, InsertPoint
    from xdsl.utils import marimo as xmo
    from xdsl.dialects.arith import AddiOp
    from xdsl.dialects.builtin import ModuleOp, IntegerType, IntegerAttr
    from xdsl.dialects.arith import ConstantOp
    return (
        AddiOp,
        Builder,
        ConstantOp,
        InsertPoint,
        IntegerAttr,
        IntegerType,
        ModuleOp,
        mo,
        xmo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Builders""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Builders are used to insert new operations in an existing block. While they also exist in MLIR, they work a bit differently in xDSL. In MLIR, builders are used to create operations, while in xDSL builders are used to insert already created operations.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Operation Constructors""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In xDSL, ops often define a custom `__init__` constructor to make operation creation easier.

    For example, here is the API for creating an `arith.constant` instance:
    """
    )
    return


@app.cell
def _(ConstantOp, IntegerAttr):
    c0 = ConstantOp(IntegerAttr(0, 64))
    c0
    return (c0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Similarly, here is the API for creating an `arith.addi` instance:""")
    return


@app.cell
def _(AddiOp, c0):
    addi = AddiOp(c0.result, c0.result)
    addi
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Generally, it is recommended to look at the definition of the operation you would like to create to see the API. Here is the xDSL generated documentation for the [`arith` dialect](https://docs.xdsl.dev/reference/dialects/arith/) and the [`scf` dialect](https://docs.xdsl.dev/reference/dialects/scf/).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using the `Builder`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A `Builder` insert new operations at a given insertion point (A position before an operation, or at the end of a block). Whenever a builder inserts a new operation, it updates its insertion point to be after the inserted operation.

    The `Builder` constructor takes a single argument, which is an `InsertPoint`. An `InsertPoint` can be created using the static methods `before`, `after`, `at_start`, and `at_end`.

    Let's look at the following program:
    """
    )
    return


@app.cell(hide_code=True)
def _(ConstantOp, IntegerAttr, IntegerType, ModuleOp, xmo):
    module = ModuleOp([
        ConstantOp(IntegerAttr(0, IntegerType(64))),
        ConstantOp(IntegerAttr(1, IntegerType(64))),
        ConstantOp(IntegerAttr(2, IntegerType(64))),
    ])

    xmo.module_html(module)
    return (module,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""The following code inserts the constant `3` and `4` in the IR using the builders.""")
    return


@app.cell
def _(Builder, ConstantOp, InsertPoint, IntegerAttr, IntegerType, module):
    # Clone the module to only do modifications locally.
    # Otherwise this messes up other cells using `module`.
    module_cloned = module.clone()
    block_cloned = module_cloned.body.block
    cst0_cloned = module_cloned.body.ops.first
    cst1_cloned = cst0_cloned.next_op

    # Insert 3 between 0 and 1
    builder1 = Builder(InsertPoint.before(cst1_cloned))
    builder1.insert(ConstantOp(IntegerAttr(3, IntegerType(64))))

    # Insert 4 and 5 at the end of the block
    builder2 = Builder(InsertPoint.at_end(block_cloned))
    builder2.insert(ConstantOp(IntegerAttr(4, IntegerType(64))))
    builder2.insert(ConstantOp(IntegerAttr(5, IntegerType(64))))

    None
    return (module_cloned,)


@app.cell
def _(module_cloned):
    module_cloned
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Small exercise

    As a simple task, insert the program so the operations are read in order `0, 1, 2, 3, 4, 5, 6, 7`
    """
    )
    return


@app.cell(hide_code=True)
def _(ConstantOp, IntegerAttr, IntegerType, ModuleOp, xmo):
    module2 = ModuleOp([
        ConstantOp(IntegerAttr(1, IntegerType(64))),
        ConstantOp(IntegerAttr(4, IntegerType(64))),
        ConstantOp(IntegerAttr(6, IntegerType(64))),
    ])

    xmo.module_html(module2)
    return (module2,)


@app.cell
def _(module2):
    module2_cloned = module2.clone()

    block = module2_cloned.body.block
    cst1 = block.first_op
    cst4 = cst1.next_op
    cst6 = cst4.next_op

    # Use the builder here
    return


if __name__ == "__main__":
    app.run()
