# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xdsl",
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 1. MLIR IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Textual Format""")
    return


@app.cell(hide_code=True)
def _(mo, triangle_text):
    mo.md(
        rf"""
    MLIR and xDSL use an encoding of the IR as a textual format for debugging, testing, and storing intermediate representations of programs.
    It can be very useful to take a program at some stage of compilation, and inspect it.
    The textual format makes this easy to do.
    Let's look at a representation of a function that sums the first `n` integers:

    {mo.ui.code_editor(triangle_text, language="javascript", disabled=True)}

    We'll look at it more in detail, but let's take a look at some broad properties:

    1. The IR is "structured".

    There are curly braces (`{{}}`) with indented code in them, a bit like the C family languages.

    2. There are assignments with `=`

    One important detail is that each value is assigned to exactly once.
    MLIR IR is in [SSA form](https://en.wikipedia.org/wiki/Static_single-assignment_form), a property that makes it easier to determine the contents of a value when reasoning about code.

    3. The things immediately to the right of the assignments are in the form `first.second`

    These things are the names of operations.
    These operations are the core building blocks of MLIR IR, and their structure and meaning is indicated by this name.
    The names are all in two parts, where the first part is the name of a dialect, a kind of namespace for related concepts, and the second makes the operation name unique within the dialect.

    With this in mind, let's zoom in to the first operation.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `builtin.module`""")
    return


@app.cell(hide_code=True)
def _(builtin, mo):
    mo.md(
        rf"""
    A module is a unit of code in xDSL and MLIR.
    It is an operation in the [`builtin` dialect](https://mlir.llvm.org/docs/Dialects/Builtin/), and holds a single _region_.

    The IR structure is a tree, with operations that have an ordered doubly-linked list of regions, each of which has a doubly-linked list of blocks, each of which has a doubly-linked list of operations, and so on.
    The smallest possible piece of code in MLIR IR is an empty module:

    ```
    {str(builtin.ModuleOp([]))}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generic Format""")
    return


@app.cell(hide_code=True)
def _(builtin, mo, print_generic):
    mo.md(
        rf"""
    The IRs above are in what's called the _custom format_, a format that allows functions to specify a pretty and concise representation.
    The _generic format_ is a more uniform and verbose representation that unambiguously shows the structure of an operation.
    Here is the above minimal module in generic format:

    ```
    {print_generic(builtin.ModuleOp([]))}
    ```

    There is a bit more going on.

    First the name of the operation is now in quotes:

    ```
    "builtin.module"
    ```

    Next is the list of operands, which is empty:

    ```
                    ()
    ```

    Then follows the region (in parentheses), with a single empty block:

    ```
                       ({{
    ^0:
    )}}
    ```

    We'll discuss blocks a bit more when we look at the `func.func` and `scf.for` operations.

    The type of the operation is always printed in the generic format, even if is an operation with no operands or results, as in this case:

    ```
        : () -> ()
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Attributes""")
    return


@app.cell(hide_code=True)
def _(builtin, mo):
    mo.md(
        rf"""
    Attributes hold compile-time data, such as constants, types, and other information.
    The IR above contains four attributes: `@triangle`, `0`, `1` and `index`.
    `index` is the type of integer values that fit in a register on the target.
    As the IR here is independent of the machine on which it will run, we don't yet specify the bitwidth of the integer.
    In MLIR, the common way to represent integers of 16, 32, 64, or other bitwidths is `i16`, `i32`, `i64`, etc.
    `@triangle` is a symbol name, denoting the name of the function.

    There are many other attributes, and many of them are in the `builtin` dialect.
    We will look into defining those in a later notebook, and will only be using attributes from the builtin dialect in this one.

    One important attribute is the dictionary attribute, which looks like this:

    ```
    {builtin.DictionaryAttr({
        "string": builtin.StringAttr("my_string"),
        "int": builtin.IntegerAttr(42, builtin.IndexType()),
        "float": builtin.FloatAttr(3.1415, builtin.f32),
        "unit": builtin.UnitAttr()
    })}
    ```

    The last entry denotes a key-value pair where the value is the unit attribute, which is omitted.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Attribute Dictionaries""")
    return


@app.cell(hide_code=True)
def _(builtin, mo):
    _module_op = builtin.ModuleOp([])
    _module_op.attributes = {"my_key": builtin.StringAttr("my_value")}

    mo.md(
        rf"""
    Operations can be supplemented with arbitrary information via their attribute dictionary.

    Here's a module with some extra information:
    ```
    {str(_module_op)}
    ```

    There are two changes here, the added dictionary and the `attributes` keyword, which is added to avoid the ambiguity between the dictionary and the region, which are both delimited with `{{}}`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The `func` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    _func_op = None
    _func_op_with_args = None
    _func_op_with_return_value = None
    _func_op_calling_example = None

    mo.md(
        rf"""
        The `func` dialect contains building blocks to model functions and function calls. It contains the following important operations:

        - **`func.func`**: This operation is used to model function definition. They contain the symbolic name of the function to be defined, along with an inner region representing the body of the function.
            ```
            {str(_func_op)}
            ```
            In order to model function parameters, the entry block of the body region has **block arguments** corresponding to each function argument. In the context of `func.func`, these arguments represent values that will be filled by the caller. For readability, the custom format of `func.func` prints them next to the function name.
            ```
            {str(_func_op_with_args)}
            ```

        - **`func.return`**: This operation represents a return statement, taking as parameters the values that should be returned. `func.return` is a terminator, meaning that it must be the last operation in its block.
            ```
            {str(_func_op_with_return_value)}
            ```

        - **`func.call`**: This operation allows calling a function by its symbol name. `func.call` takes as operands the values of the function parameters, and its results are the return values of the function.
            ```
            {str(_func_op_calling_example)}
            ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Properties""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exercise 1: Your First Function""")
    return


@app.cell(hide_code=True)
def _():
    first_text = """\
    func.func @first(%arg0: index, %arg1: index) -> index {
        // Change this to return the second argument instead
        func.return %arg0 : index
    }\
    """
    return (first_text,)


@app.cell(hide_code=True)
def _(first_text, mo):
    first_text_area = mo.ui.code_editor(first_text, language="javascript")
    return (first_text_area,)


@app.cell(hide_code=True)
def _(Parser, ctx, first_text_area, run_func):
    first_error_text = ""
    first_results_12_text = ""
    first_results_34_text = ""
    try:
        first_module = Parser(ctx, first_text_area.value).parse_module()
        first_results_12 = run_func(first_module, "first", (1, 2))
        first_results_34 = run_func(first_module, "first", (3, 4))
        first_results_12_text = f"first(1, 2) = {first_results_12}"
        first_results_34_text = f"first(3, 4) = {first_results_34}"
    except Exception as e:
        error_text = str(e)
    if first_error_text:
        first_info_text = f"""
        Error:

        ```
        {first_error_text}
        ```
        """
    else:
        first_info_text = f"""\
        Here are the outputs when running the function with inputs (1, 2), and (3, 4):

        ```
        {first_results_12_text}
        {first_results_34_text}
        ```
        """
    return (
        error_text,
        first_error_text,
        first_info_text,
        first_module,
        first_results_12,
        first_results_12_text,
        first_results_34,
        first_results_34_text,
    )


@app.cell(hide_code=True)
def _(first_info_text, first_text_area, mo):
    mo.vstack((first_text_area, mo.md(first_info_text)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The `scf` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    _if_op = None
    _if_op_with_else = None
    _if_op_with_yield = None

    mo.md(
        rf"""
        The `scf` dialect contains building blocks to model Structured Control-Flow (SCF). In contrast to LLVM-like Control-Flow Graphs (CFG), Structured Control-Flow is a model of control-flow based on regions. This model of control-flow is similar in many ways to the ones in imperative languages. It contains the following important operations:

        - **`scf.if`**: This operation represents a an if-statement. It takes in a boolean value, and if that value is true, it steps inside its inner region (the "then" region), skipping it otherwise.
            ```
            {str(_if_op)}
            ```
            An additional region can be added (the "else" region) that is stepped inside only when the boolean value is false.
            ```
            {str(_if_op_with_else)}
            ```
            If two regions are specified, `scf.if` can have result values of which the value is defined differently in each region. This feature will be presented with the next operation.

        - **`scf.yield`**: This operation is a terminator allowing to yield values from SCF constructs. For example, in the context of an `scf.if`, one may want to declare a single value `%res` that has different content depending on which branch of the `scf.if` is taken. In order to do this, one can add `%res` as a result value to the `scf.if`. Then, `scf.yield` is used in each of the regions to define the content of `%res`.
            ```
            {str(_if_op_with_yield)}
            ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Block Arguments""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exercise 2: Factorial Function""")
    return


@app.cell(hide_code=True)
def _(mo, triangle_text):
    second_input_text = mo.ui.text("5")
    second_text_area = mo.ui.code_editor(
        triangle_text.replace("triangle", "second"), language="javascript"
    )
    return second_input_text, second_text_area


@app.cell(hide_code=True)
def _(Parser, ctx, run_func, second_input_text, second_text_area):
    second_error_text = ""
    second_results_text = ""
    try:
        second_input = int(second_input_text.value)
        second_inputs = (second_input,)
        second_module = Parser(ctx, second_text_area.value).parse_module()
        second_results = run_func(second_module, "second", second_inputs)
        second_results_text = f"second({second_input}) = {second_results}"
    except Exception as e:
        print("no")
        second_error_text = str(e)

    if second_error_text:
        second_info_text = f"""
        Error:

        ```
        {second_error_text}
        ```
        """
    else:
        second_info_text = f"""\
        Change the definition of `second` to compute the factorial of the input.
        Assume that the input is non-negative.

        ```
        {second_results_text}
        ```
        """
    return (
        second_error_text,
        second_info_text,
        second_input,
        second_inputs,
        second_module,
        second_results,
        second_results_text,
    )


@app.cell(hide_code=True)
def _(mo, second_info_text, second_input_text, second_text_area):
    mo.vstack(
        (
            mo.md(f"Input: {second_input_text}"),
            second_text_area,
            mo.md(second_info_text),
        )
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(ModuleOp, StringIO):
    from xdsl.printer import Printer

    def print_generic(module: ModuleOp) -> str:
        io = StringIO()
        Printer(io, print_generic_format=True).print(module)
        return io.getvalue()

    return Printer, print_generic


@app.cell(hide_code=True)
def _():
    # import relevant dialects and load them in context

    from xdsl.context import MLContext
    from xdsl.dialects import builtin, func, scf, arith

    ctx = MLContext()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(arith.Arith)
    return MLContext, arith, builtin, ctx, func, scf


@app.cell(hide_code=True)
def _():
    triangle_text = """\
    builtin.module {
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
      }
    }\
    """
    return (triangle_text,)


@app.cell(hide_code=True)
def _(ctx, triangle_text):
    # parse triangle module to make sure it's valid IR

    from io import StringIO

    from xdsl.parser import Parser, Input

    triangle_module = Parser(ctx, triangle_text, "").parse_module()
    # triangle_module
    return Input, Parser, StringIO, triangle_module


@app.cell(hide_code=True)
def _(ModuleOp):
    from typing import Any

    def run_func(module: ModuleOp, name: str, args: tuple[Any, ...]):
        from xdsl.interpreter import Interpreter
        from xdsl.interpreters import scf, arith, func

        interpreter = Interpreter(module)
        interpreter.register_implementations(scf.ScfFunctions)
        interpreter.register_implementations(arith.ArithFunctions)
        interpreter.register_implementations(func.FuncFunctions)

        res = interpreter.call_op(name, args)

        if len(res) == 1:
            res = res[0]

        return res

    return Any, run_func


if __name__ == "__main__":
    app.run()
