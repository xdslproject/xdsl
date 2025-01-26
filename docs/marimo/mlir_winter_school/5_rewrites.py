import marimo

__generated_with = "0.10.17"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, GreedyRewritePatternApplier, PatternRewriteWalker
    from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.dialects.func import FuncOp
    return (
        AddiOp,
        ConstantOp,
        FuncOp,
        GreedyRewritePatternApplier,
        ModuleOp,
        MuliOp,
        PatternRewriteWalker,
        PatternRewriter,
        RewritePattern,
        mo,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Pattern rewrites

        ## Rationale

        A pattern rewrite is a compiler transformation that matches a DAG in the IR, and replace it with another DAG. For instance, simplifying `x + 0` to `x` is a common optimization that is represented as a pattern rewrite.

        As xDSL and MLIR allow the definition of higher-level dialects, pattern rewrites are very common, and can be used to both write high-level optimizations, and lowerings from a high-level dialect to a lower-level one. A general rationale for pattern rewrites in MLIR can be found [here](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/).

        Pattern rewrites are applied step by step on the IR. For instance, using the rewrite `x + 0 -> x`, the IR `x + 0 + 0` will be progressively transformed to `x + 0`, then `x using this single rewrite. Different application ordering and variations exist for rewrite patterns, which will be covered by Matthias Springer at 4pm on day 3 of the winter school.

        ## Defining a pattern rewrite

        Each Pattern rewrite is a class that inherit from `PatternRewrite`. It defines a single `match_and_rewrite` method, that is called to apply a pattern on an operation. The method will either return without any modification of the IR, or will modify the IR using the `Rewriter`. The most common operation to call on the `Rewriter` is `replace_matched_op`, which will replace the matched operation with a sequence of other operations, and replace the result values of the matched operation with new values.

        Here are two examples of pattern for `x + 0 -> x` and `x * 2 -> x + x`:
        """
    )
    return


@app.cell
def _(
    AddiOp,
    ConstantOp,
    CostantOp,
    MuliOp,
    Operation,
    RewritePattern,
    Rewriter,
):
    # The rewrite x + 0 -> x
    class AddZeroPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: Rewriter):
            # Match an `arith.addi`
            if not isinstance(op, AddiOp):
                return

            # Check if the right hand side is a constant
            if not isinstance(cst := op.rhs.owner, ConstantOp):
                return

            # Check if the constant is 0
            if cst.value.value.data != 0:
                return

            # Replace `x + 0` with `x`
            # The first argument is the new operations to insert
            # The second argument is the replacement for the results of the matched
            # operation (op). As we replace `x + 0` with `x`, we replace the
            # `arith.addi` results with its first argument.
            rewriter.replace_matched_op([], new_results=[op.lhs])

    # The rewrite x * 2 -> x + x
    class MulTwoPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: Rewriter):
            # Match an `arith.addi`
            if not isinstance(op, MuliOp):
                return
            x = op.lhs

            # Check if the right hand side is a constant
            if not isinstance(cst := op.rhs.owner, CostantOp):
                return

            # Check if the constant is 2
            if cst.value.value.data != 2:
                return

            # Replace `x * 2` with `x + x`
            # The results of the `arith.muli` operation are by default the results of the
            # last operation added, so here the `arith.addi`
            add = AddiOp(x, x)
            rewriter.replace_matched_op([add])
    return AddZeroPattern, MulTwoPattern


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Applying rewrite patterns

        There are two steps to apply rewrite patterns in xDSL.

        * First, combining multiple rewrite patterns in a single one, giving priorities between patterns.
        * Then, walking the IR and applying the pattern recursively on all operations until no more pattern can be applied on any operation:

        Here is an example on how to apply rewrites on all operations in a module:
        """
    )
    return


@app.cell
def _(
    AddZeroPattern,
    GreedyRewritePatternApplier,
    ModuleOp,
    MulTwoPattern,
    Operation,
    PatternRewriteWalker,
):
    def apply_all_rewrites(module: ModuleOp, rewrites: list[Operation]):
        merged_pattern = GreedyRewritePatternApplier(AddZeroPattern(), MulTwoPattern())
        walker = PatternRewriteWalker(merged_pattern)
        walker.rewrite_module(module)
    return (apply_all_rewrites,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Before/After applying rewrite patterns

        Here are some examples of programs before and after applying optimzation rewrite patterns:
        """
    )
    return


if __name__ == "__main__":
    app.run()
