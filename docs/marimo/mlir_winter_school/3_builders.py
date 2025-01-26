import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3. Builders""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Builders are used to insert new operations in an existing block. While they also exist in MLIR, they work a bit differently in xDSl. In MLIR, builders are used to create operations, while in xDSL builders are used to insert already created operations.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Insertion Point

        Insertion points are defined by `InsertPoint`, and correspond to a location in a block. They are either pointing to the place before an operation, or at the end of a block. An `InsertPoint` can be created using the static methods `before`, `after`, `at_start`, and `at_end`.

        Here is how to define an `InsertPoint` for the following program:
        """
    )
    return


@app.cell(hide_code=True)
def _(xmo):
    from xdsl.dialects.builtin import ModuleOp, IntegerType, IntegerAttr
    from xdsl.dialects.arith import ConstantOp

    module = ModuleOp([
        ConstantOp(IntegerAttr(0, IntegerType(64))),
        ConstantOp(IntegerAttr(1, IntegerType(64))),
        ConstantOp(IntegerAttr(2, IntegerType(64))),
    ])

    xmo.module_html(module)
    return ConstantOp, IntegerAttr, IntegerType, ModuleOp, module


@app.cell
def _(module):
    from xdsl.builder import Builder, InsertPoint

    # The module only block
    block = module.body.block

    # The three constant operations
    cst0 = module.body.ops.first
    cst1 = cst0.next_op
    cst2 = cst1.next_op

    # The point between cst0 and cst1
    _ = InsertPoint.before(cst1)

    # The point between cst1 and cst2
    _ = InsertPoint.after(cst1)

    # The point at the end of the block, after cst2
    _ = InsertPoint.at_end(block)

    # The point at the begining of the block, before cst0
    _ = InsertPoint.at_start(block)
    return Builder, InsertPoint, block, cst0, cst1, cst2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Builders

        Builders insert new operations at their insertion point. Here is an example on how to insert a new operation in a block:
        """
    )
    return


@app.cell(hide_code=True)
def _(module, xmo):
    xmo.module_html(module)
    return


@app.cell
def _(
    Builder,
    ConstantOp,
    InsertPoint,
    IntegerAttr,
    IntegerType,
    module,
    xmo,
):
    # Clone the module to only do modifications locally.
    # Otherwise this messes up other cells using `module`.
    module_cloned = module.clone()
    block_cloned = module_cloned.body.block
    cst0_cloned = module_cloned.body.ops.first
    cst1_cloned = cst0_cloned.next_op
    cst2_cloned = cst1_cloned.next_op


    # Create a new builder at the location before the constant 1.
    builder = Builder(InsertPoint.before(cst1_cloned))

    # Insert a new operation at the builder location.
    builder.insert(ConstantOp(IntegerAttr(42, IntegerType(32))))

    # Change the builder insertion point.
    # This is done by modifying the `insertion_point` field.
    builder.insertion_point = InsertPoint.at_end(block_cloned)

    # Insert a new operation at the builder location.
    builder.insert(ConstantOp(IntegerAttr(1337, IntegerType(32))))

    xmo.module_html(module_cloned)
    return (
        block_cloned,
        builder,
        cst0_cloned,
        cst1_cloned,
        cst2_cloned,
        module_cloned,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from xdsl.utils import marimo as xmo
    return (xmo,)


if __name__ == "__main__":
    app.run()
