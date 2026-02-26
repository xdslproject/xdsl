import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from xdsl.utils import marimo as xmo
    return (xmo,)


@app.cell(hide_code=True)
def _():
    from xdsl.dialects import arith, builtin, pdl, func
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.transforms.apply_pdl import ApplyPDLPass
    from xdsl.transforms.dead_code_elimination import dce
    from xdsl.rewriter import Rewriter
    from xdsl.builder import InsertPoint

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(pdl.PDL)
    ctx.load_dialect(func.Func)
    return (
        ApplyPDLPass,
        InsertPoint,
        Parser,
        Rewriter,
        arith,
        builtin,
        ctx,
        dce,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Pattern Description Language (PDL)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    One of the most frequent kinds of transformations on intermediate representations in compilers are local, "peephole", rewrites, which transform an operation by inspecting its local context only.
    An example of this might be rewriting `arith.add` into `llvm.add`, which can be done just by inspecting the type of the operation, or local optimisations such as rewriting an `arith.add` with `0` to the other operand.
    In both MLIR and xDSL these can be written using the native language API (C++ or Python), a flexible and productive approach with two important flaws: reasoning about these rewrites requires reasoning about the semantics of the host language, which is famously difficult for both C++ and Python, and generating them is much less convenient than generating MLIR IR directly.
    The `pdl` dialect addresses both of these issues.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PDL Patterns""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First let's look at the pattern that rewrites `first` to `second`:""")
    return


@app.cell(hide_code=True)
def _(Parser, builtin, ctx, xmo):
    first_text = """
    func.func @first(%x : i32) -> i32 {
      %c0 = arith.constant 0 : i32
      %y = arith.muli %x, %c0 : i32
      func.return %y : i32
    }
    """

    second_text = """
    func.func @second(%x : i32) -> i32 {
      %c0 = arith.constant 0 : i32
      func.return %c0 : i32
    }
    """

    first_op = Parser(ctx, first_text).parse_op()
    second_op = Parser(ctx, second_text).parse_op()

    xmo.module_html(builtin.ModuleOp([first_op.clone(), second_op.clone()]))
    return first_op, second_op


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that these are equivalent, as `x * 0 = 0` for all `x`.

    The following pattern implements the rewrite for 32-bit integers:
    """
    )
    return


@app.cell(hide_code=True)
def _(Parser, ctx, xmo):
    times_zero_text = """
    pdl.pattern @x_times_zero : benefit(2) {
      %t = pdl.type
      %x = pdl.operand
      %c0_attr = pdl.attribute = 0 : i32
      %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
      %c0_res = pdl.result 0 of %c0_op
      %x_times_zero_op = pdl.operation "arith.muli" (%x, %c0_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
      pdl.rewrite %x_times_zero_op {
        pdl.replace %x_times_zero_op with (%c0_res : !pdl.value)
      }
    }
    """

    times_zero_op = Parser(ctx, times_zero_text).parse_op()

    xmo.module_html(times_zero_op)
    return (times_zero_op,)


@app.cell(hide_code=True)
def _(ApplyPDLPass, builtin, ctx, first_op, second_op, times_zero_op):
    def test_rewrite():
        input_copy = first_op.clone()
        input_copy.sym_name = builtin.StringAttr("second")
        pattern_copy = times_zero_op.clone()
        module = builtin.ModuleOp([input_copy, pattern_copy])
        ApplyPDLPass().apply(ctx, module)
        assert str(input_copy) == str(second_op)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's look at the pattern in more detail.

    The [`pdl.pattern` operation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlpattern-pdlpatternop) has an optional name, a "benefit" field, and a body.

    ```
    pdl.pattern : benefit(2) {
      ...
    }
    ```

    The non-negative benefit is used in the case where multiple patterns match the same program.

    The body consists of two parts, a declarative matching region terminated with a [`pdl.rewrite` operation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrewrite-pdlrewriteop) containing an imperative rewrite region:

    ``` C
    // Match old IR
    pdl.rewrite %root {
      // Create new IR
    }
    ```

    The rewrite takes a number of arguments which correspond to the operation being rewritten.
    In practice, there is almost always one operation being matched on.

    Here's the matching region with comments:

    ```
    // 1. The type that we're operating on
    %t = pdl.type
    // 2. The lhs operand to the operation
    %x = pdl.operand
    // 3. The zero attribute of our target type
    %c0_attr = pdl.attribute = 0 : i32
    // 4. A specification of the constant operation we want to match on
    %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
    // 5. The result value of the constant operation
    %c0_res = pdl.result 0 of %c0_op
    // 6. Finally, the operation that we want to rewrite
    %x_times_zero_op = pdl.operation "arith.muli" (%x, %c0_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    ```

    Note that in this region, we don't have to provide all of the components of the matched operation, instead we only have to provide the constraints that we care about.

    This example's rewrite operation is quite simple, replacing the matched operation with the zero operand:

    ```
    pdl.rewrite %x_times_zero_op {
      pdl.replace %x_times_zero_op with (%c0_res : !pdl.value)
    }
    ```

    Please refer to the [dialect reference](https://mlir.llvm.org/docs/Dialects/PDLOps/) for the full list of operations and types.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, write_check):
    mo.md(
        rf"""
    ## Exercises {write_check}

    In this notebook, you'll write patterns in PDL to transform this input function:
    """
    )
    return


@app.cell(hide_code=True)
def _(arith, builtin):
    from xdsl.frontend.pyast.context import PyASTContext
    from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

    # Set up the AST parsing context
    pyast_ctx = PyASTContext()
    pyast_ctx.register_type(float, builtin.f64)
    pyast_ctx.register_function(float.__add__, arith.AddfOp)
    pyast_ctx.register_function(float.__sub__, arith.SubfOp)
    pyast_ctx.register_function(float.__mul__, arith.MulfOp)
    pyast_ctx.register_function(float.__truediv__, arith.DivfOp)
    return (pyast_ctx,)


@app.cell
def _(mo, pyast_ctx):
    @pyast_ctx.parse_program
    def main(a: float, b: float, c: float) -> float:
        return (c + (a - a)) / (b / b)

    import inspect
    lines = inspect.getsource(main)
    mo.md(f"```python\n{lines}```")
    return (main,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As you can see, it just returns `c`, we'll want to rewrite our program to return `c` directly, without the useless computations.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is the corresponding IR:""")
    return


@app.cell(hide_code=True)
def _(main, xmo):
    xmo.module_html(main.module.body.ops.first)
    return


@app.cell(hide_code=True)
def _(mo):
    x_minus_x_text = """\
    pdl.pattern @x_minus_x : benefit(2) {
      %t = pdl.type
      %x = pdl.operand

      // Add op to match here

      // Uncomment these lines:
      // pdl.rewrite %x_minus_x {
      //   %c0_attr = pdl.attribute = 0.0 : f64
      //   %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
      //   %c0_res = pdl.result 0 of %c0_op
      //   pdl.replace %x_minus_x with (%c0_res : !pdl.value)
      // }
    }"""

    x_minus_x_text_area = mo.ui.code_editor(x_minus_x_text, language="javascript")
    return (x_minus_x_text_area,)


@app.cell(hide_code=True)
def _(info_text, mo, x_minus_x_text_area):
    mo.vstack((
        mo.md("### Exercise 1\nFill out the matching region below to implement `x - x -> 0`:"),
        x_minus_x_text_area,
        mo.md(info_text))
    )
    return


@app.cell(hide_code=True)
def _(mo, xmo):
    x_minus_x_text_solution = """\
    pdl.pattern @x_minus_x : benefit(2) {
      %t = pdl.type
      %x = pdl.operand
      %x_minus_x = pdl.operation "arith.subf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
      pdl.rewrite %x_minus_x {
        %c0_attr = pdl.attribute = 0.0 : f64
        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
        %c0_res = pdl.result 0 of %c0_op
        pdl.replace %x_minus_x with (%c0_res : !pdl.value)
      }
    }"""

    mo.accordion({"Solution": xmo.module_html(x_minus_x_text_solution)})
    return (x_minus_x_text_solution,)


@app.cell(hide_code=True)
def _(mo):
    x_plus_zero_text = """\
    pdl.pattern @x_plus_zero : benefit(2) {
      %t = pdl.type
      %x = pdl.operand

      // Implement matching ops here
      // %c0_attr =
      // %c0_op =
      // %c0_res = pdl.result 0 of %c0_op
      // %x_plus_zero_op =

      // Uncomment these lines:
      // pdl.rewrite %x_plus_zero_op {
      //   pdl.replace %x_plus_zero_op with (%x : !pdl.value)
      // }
    }"""
    x_plus_zero_text_area = mo.ui.code_editor(x_plus_zero_text, language="javascript")
    return (x_plus_zero_text_area,)


@app.cell(hide_code=True)
def _(info_text, mo, x_plus_zero_text_area):
    mo.vstack((
        mo.md("### Exercise 2\nFill out the matching region below to implement `x + 0 -> x`:"),
        x_plus_zero_text_area,
        mo.md(info_text))
    )
    return


@app.cell(hide_code=True)
def _(mo, xmo):
    x_plus_zero_text_solution = """\
    pdl.pattern @x_plus_zero : benefit(2) {
      %t = pdl.type
      %x = pdl.operand
      %c0_attr = pdl.attribute = 0.0 : f64
      %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
      %c0_res = pdl.result 0 of %c0_op
      %x_plus_zero_op = pdl.operation "arith.addf" (%x, %c0_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
      pdl.rewrite %x_plus_zero_op {
        pdl.replace %x_plus_zero_op with (%x : !pdl.value)
      }
    }"""

    mo.accordion({"Solution": xmo.module_html(x_plus_zero_text_solution)})
    return (x_plus_zero_text_solution,)


@app.cell(hide_code=True)
def _(mo):
    x_div_x_text = """\
    pdl.pattern @x_div_x : benefit(2) {
      %t = pdl.type
      %x = pdl.operand
      %x_div_x_op = pdl.operation "arith.divf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
      pdl.rewrite %x_div_x_op {
        // Implement rewrite region here
      }
    }"""
    x_div_x_text_area = mo.ui.code_editor(x_div_x_text, language="javascript")
    return (x_div_x_text_area,)


@app.cell(hide_code=True)
def _(info_text, mo, x_div_x_text_area):
    mo.vstack((
        mo.md("### Exercise 3\nFill out the matching region to implement `x / x -> 1`:"),
        x_div_x_text_area,
        mo.md(info_text))
    )
    return


@app.cell(hide_code=True)
def _(mo, xmo):
    x_div_x_text_solution = """\
    pdl.pattern @x_div_x : benefit(2) {
      %t = pdl.type
      %x = pdl.operand
      %x_div_x_op = pdl.operation "arith.divf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
      pdl.rewrite %x_div_x_op {
        %c1_attr = pdl.attribute = 1.0 : f64
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%t : !pdl.type)
        %c1_res = pdl.result 0 of %c1_op
        pdl.replace %x_div_x_op with (%c1_res : !pdl.value)
      }
    }"""

    mo.accordion({"Solution": xmo.module_html(x_div_x_text_solution)})
    return (x_div_x_text_solution,)


@app.cell(hide_code=True)
def _(mo):
    x_div_one_text = """\
    pdl.pattern @x_div_one : benefit(2) {
      %t = pdl.type
      %x = pdl.operand

      // Implement matching ops here
      // %c1_attr =
      // %c1_op =
      // %c1_res = pdl.result 0 of %c1_op
      // %x_div_one_op =

      // Uncomment these lines:
      // pdl.rewrite %x_div_one_op {
      //   pdl.replace %x_div_one_op with (%x : !pdl.value)
      // }
    }"""
    x_div_one_text_area = mo.ui.code_editor(x_div_one_text, language="javascript")
    return (x_div_one_text_area,)


@app.cell(hide_code=True)
def _(info_text, mo, x_div_one_text_area):
    mo.vstack((
        mo.md("### Exercise 4\nFill out the matching region below to implement `x / 1 -> x`:"),
        x_div_one_text_area,
        mo.md(info_text))
    )
    return


@app.cell(hide_code=True)
def _(mo, xmo):
    x_div_one_text_solution = """\
    pdl.pattern @x_div_one : benefit(2) {
      %t = pdl.type
      %x = pdl.operand
      %c1_attr = pdl.attribute = 1.0 : f64
      %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%t : !pdl.type)
      %c1_res = pdl.result 0 of %c1_op
      %x_div_one_op = pdl.operation "arith.divf" (%x, %c1_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
      pdl.rewrite %x_div_one_op {
        pdl.replace %x_div_one_op with (%x : !pdl.value)
      }
    }"""

    mo.accordion({"Solution": xmo.module_html(x_div_one_text_solution)})
    return (x_div_one_text_solution,)


@app.cell
def _():
    expected_text = """\
    func.func @main(%a : f64, %b : f64, %c : f64) -> f64 {
      func.return %c : f64
    }"""
    return (expected_text,)


@app.cell
def _(
    ApplyPDLPass,
    InsertPoint,
    Parser,
    Rewriter,
    ctx,
    dce,
    expected_text,
    main,
    x_div_one_text_area,
    x_div_x_text_area,
    x_minus_x_text_area,
    x_plus_zero_text_area,
):
    _error_text = ""
    _results_text = ""
    try:
        _patterns_text = x_minus_x_text_area.value + x_plus_zero_text_area.value + x_div_x_text_area.value + x_div_one_text_area.value
        _module = Parser(ctx, _patterns_text).parse_module()
        _cloned_func = main.module.body.ops.first.clone()
        Rewriter.insert_op(_cloned_func, InsertPoint.at_start(_module.body.block))
        ApplyPDLPass().apply(ctx, _module)
        dce(_module)
        _results_text = str(_cloned_func)
        write_check = "✅" if _results_text == expected_text else "❌"
        if _results_text == expected_text:
            check_text = f"{write_check} Applying all patterns yields expected function."
        else:
            check_text = f"{write_check} Applying all patterns does not yield expected function."
    except Exception as e:
        _error_text = str(e)
    if _error_text:
        info_text = f"""
    /// attention | Error:

    ```
    {_error_text}
    ```
    """
    else:
        info_text = f"""
    Here are the outputs when running the function:

    ```
    {_results_text}
    ```

    {check_text}
    """
    return info_text, write_check


@app.cell(hide_code=True)
def _(
    ApplyPDLPass,
    InsertPoint,
    Parser,
    Rewriter,
    ctx,
    dce,
    expected_text,
    main,
    x_div_one_text_solution,
    x_div_x_text_solution,
    x_minus_x_text_solution,
    x_plus_zero_text_solution,
):
    def test_solutions():
        _solutions_text = x_minus_x_text_solution + x_plus_zero_text_solution + x_div_x_text_solution + x_div_one_text_solution
        _module = Parser(ctx, _solutions_text).parse_module()
        _cloned_func = main.module.body.ops.first.clone()
        Rewriter.insert_op(_cloned_func, InsertPoint.at_start(_module.body.block))
        ApplyPDLPass().apply(ctx, _module)
        dce(_module)
        _results_text = str(_cloned_func)
        assert _results_text == expected_text
    return


if __name__ == "__main__":
    app.run()
