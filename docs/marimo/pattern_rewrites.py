# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xdsl==0.27.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from xdsl.utils import marimo as xmo

    from xdsl.parser import Parser
    from xdsl.context import Context

    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.func import Func

    from xdsl.ir import Operation

    from xdsl.rewriter import Rewriter
    from xdsl.pattern_rewriter import (
        RewritePattern,
        GreedyRewritePatternApplier,
        PatternRewriteWalker,
    )
    from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp
    from xdsl.dialects.builtin import ModuleOp
    return (
        AddiOp,
        Arith,
        Builtin,
        ConstantOp,
        Context,
        Func,
        GreedyRewritePatternApplier,
        ModuleOp,
        MuliOp,
        Operation,
        Parser,
        PatternRewriteWalker,
        RewritePattern,
        Rewriter,
        mo,
        xmo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Pattern Rewrites""")
    return


@app.cell(hide_code=True)
def _(Parser, ctx):
    _before_text = """\
    func.func @my_func(%x : index) -> index {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %also_x = arith.addi %c0, %x : index
      %two_x = arith.muli %also_x, %c2 : index
      func.return %two_x : index
    }
    """
    before_module = Parser(ctx, _before_text).parse_module()
    None
    return (before_module,)


@app.cell(hide_code=True)
def _(Parser, ctx):
    _after_text = """\
    func.func @my_func(%x : index) -> index {
      %two_x = arith.addi %x, %x : index
      func.return %two_x : index
    }
    """
    after_module = Parser(ctx, _after_text).parse_module()
    None
    return (after_module,)


@app.cell(hide_code=True)
def _(after_module, before_module, mo, xmo):
    mo.md(
        f"""
    Most of the work in compilers is rewriting IR.
    While some transformations have to take modules into account, some can be done locally, such as rewriting an addition with zero to just use the non-zero input.
    These kinds of rewrites are called pattern rewrites, as they apply if some local pattern is matched.

    We can use xDSL's pattern rewriting API to rewrite this function:

    {xmo.module_html(before_module)}

    Into this one:

    {xmo.module_html(after_module)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Rationale

    A pattern rewrite is a compiler transformation that matches a DAG in the IR, and replace it with another DAG. For instance, simplifying `x + 0` to `x` is a common optimization that is represented as a pattern rewrite.

    As xDSL and MLIR allow the definition of higher-level dialects, pattern rewrites are very common, and can be used to both write high-level optimizations, and lowerings from a high-level dialect to a lower-level one. A general rationale for pattern rewrites in MLIR can be found [here](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/).

    Pattern rewrites are applied step by step on the IR. For instance, using the rewrite `x + 0 -> x`, the IR `x + 0 + 0` will be progressively transformed to `x + 0`, and then `x` using this single rewrite. Different application ordering and variations exist for rewrite patterns,
    please see `PatternRewriteWalker` for more details.

    ## Defining a pattern rewrite

    Each Pattern rewrite is a class that inherits from `PatternRewrite`. It defines a single `match_and_rewrite` method, that is called to apply a pattern on an operation. The method will either return without any modification of the IR, or will modify the IR using the `Rewriter`. The most common operation to call on the `Rewriter` is `replace_matched_op`, which will replace the matched operation with a sequence of other operations, and replace the result values of the matched operation with new values.

    Here are two examples of pattern for `x + 0 -> x` and `x * 2 -> x + x`:
    """
    )
    return


@app.cell
def _(
    AddiOp,
    ConstantOp,
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
            rewriter.replace_op(op, [], new_results=[op.lhs])

    # The rewrite x * 2 -> x + x
    class MulTwoPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: Rewriter):
            # Match an `arith.muli`
            if not isinstance(op, MuliOp):
                return
            x = op.lhs

            # Check if the right hand side is a constant
            if not isinstance(cst := op.rhs.owner, ConstantOp):
                return

            # Check if the constant is 2
            if cst.value.value.data != 2:
                return

            # Replace `x * 2` with `x + x`
            # The results of the `arith.muli` operation are by default the results of the
            # last operation added, so here the `arith.addi`
            add = AddiOp(x, x)
            rewriter.replace_op(op, [add])
    return AddZeroPattern, MulTwoPattern


@app.cell(hide_code=True)
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
    return


@app.cell(hide_code=True)
def _(Arith, Builtin, Context, Func):
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Func)
    return (ctx,)


if __name__ == "__main__":
    app.run()
