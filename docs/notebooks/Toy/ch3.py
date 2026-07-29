import marimo

__generated_with = "0.23.6"
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
def _(xmo):
    # Depend on xmo so marimo keeps `return (xmo,)` in the sentinel cell above.
    # Docs export replaces that exact cell body (SYNC_XDSL_IMPORT).
    _ = xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 3: High-level Language-Specific Analysis and Transformation

    As we saw in the previous chapter, the IR generated from the input program has many
    opportunities for optimisation. In this chapter, we'll implement three optimisations:

    1. Removing redundant reshapes
    2. Reshaping constants during compilation time
    3. Eliminating operations whose results are not used

    Let's take a look again at our example input:
    """)
    return


@app.cell
def _():
    from toy.compiler import parse_toy
    from toy.dialects import toy

    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    from xdsl.ir import OpResult
    from xdsl.pattern_rewriter import (
        PatternRewriter,
        PatternRewriteWalker,
        RewritePattern,
        op_type_rewrite_pattern,
    )
    from xdsl.printer import Printer
    from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
    from xdsl.utils.hints import isa

    example = """
    def main() {
      var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
      var b<6> = [1, 2, 3, 4, 5, 6];
      var c<2, 3> = b;
      var d = a + c;
      print(d);
    }
    """

    module = parse_toy(example)
    Printer().print_op(module)
    print()
    return (
        DenseIntOrFPElementsAttr,
        OpResult,
        PatternRewriteWalker,
        PatternRewriter,
        Printer,
        RemoveUnusedOperations,
        RewritePattern,
        isa,
        module,
        op_type_rewrite_pattern,
        toy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Redundant Reshapes
    """)
    return


@app.cell
def _(
    OpResult,
    PatternRewriteWalker,
    PatternRewriter,
    Printer,
    RewritePattern,
    module,
    op_type_rewrite_pattern,
    toy,
):
    class ReshapeReshapeOptPattern(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: toy.ReshapeOp, rewriter: PatternRewriter):
            """
            Reshape(Reshape(x)) = Reshape(x)
            """
            # Look at the input of the current reshape.
            reshape_input = op.arg
            if not isinstance(reshape_input, OpResult):
                # Input was not produced by an operation, could be a function argument
                return

            reshape_input_op = reshape_input.op
            if not isinstance(reshape_input_op, toy.ReshapeOp):
                # Input defined by another reshape? If not, no match.
                return

            new_op = toy.ReshapeOp(reshape_input_op.arg, op.res.type)
            rewriter.replace(op, new_op)

    module_rr = module.clone()
    # Use `PatternRewriteWalker` to rewrite all matched operations
    PatternRewriteWalker(ReshapeReshapeOptPattern()).rewrite_module(module_rr)
    Printer().print_op(module_rr)
    return (module_rr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This looks very similar to what we had before, but is subtly different. Importantly,
    the reshape that assigns to %4 now takes %2 as input, instead of %3. %3 is now no longer
    used, and because it's an operation with no observable side-effects, we can avoid doing
    the work altogether.
    """)
    return


@app.cell
def _(PatternRewriteWalker, Printer, RemoveUnusedOperations, module_rr):
    module_dce = module_rr.clone()
    PatternRewriteWalker(RemoveUnusedOperations()).rewrite_module(module_dce)
    Printer().print_op(module_dce)
    return (module_dce,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fold Constant Reshaping

    One more opportunity for optimisation is to reshape the constants at compile-time,
    instead of at runtime. We can do this with another custom `RewritePattern`:
    """)
    return


@app.cell
def _(
    DenseIntOrFPElementsAttr,
    OpResult,
    PatternRewriteWalker,
    PatternRewriter,
    Printer,
    RewritePattern,
    isa,
    module_dce,
    op_type_rewrite_pattern,
    toy,
):
    class FoldConstantReshapeOptPattern(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: toy.ReshapeOp, rewriter: PatternRewriter):
            """
            Reshaping a constant can be done at compile time
            """
            # Look at the input of the current reshape.
            reshape_input = op.arg
            if not isinstance(reshape_input, OpResult):
                # Input was not produced by an operation, could be a function argument
                return

            reshape_input_op = reshape_input.op
            if not isinstance(reshape_input_op, toy.ConstantOp):
                # Input defined by a constant? If not, no match.
                return

            assert isa(op.res.type, toy.TensorTypeF64)

            new_value = DenseIntOrFPElementsAttr.from_list(
                type=op.res.type, data=reshape_input_op.value.get_values()
            )
            new_op = toy.ConstantOp(new_value)
            rewriter.replace(op, new_op)

    module_fold = module_dce.clone()
    PatternRewriteWalker(FoldConstantReshapeOptPattern()).rewrite_module(module_fold)
    Printer().print_op(module_fold)
    return (module_fold,)


@app.cell
def _(PatternRewriteWalker, Printer, RemoveUnusedOperations, module_fold):
    module_final = module_fold.clone()
    # Remove now unused original constants
    PatternRewriteWalker(RemoveUnusedOperations()).rewrite_module(module_final)
    Printer().print_op(module_final)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we've done all the optimisations we could on this level of abstraction, let's
    go one level lower towards RISC-V.
    """)
    return


if __name__ == "__main__":
    app.run()
