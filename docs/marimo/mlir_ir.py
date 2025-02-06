import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from xdsl.context import MLContext
    from xdsl.dialects import builtin, arith, func, scf
    from xdsl.utils import marimo as xmo
    from xdsl.printer import Printer
    from typing import Any
    return Any, MLContext, Printer, arith, builtin, func, mo, scf, xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MLIR IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Textual Format""")
    return


@app.cell(hide_code=True)
def _(mo, triangle_text):
    mo.md(fr"""
    MLIR and xDSL use an encoding of the IR as a textual format for debugging, testing, and storing intermediate representations of programs.
    It can be very useful to take a program at some stage of compilation, and inspect it.
    The textual format makes this easy to do.
    Let's look at a representation of a function that sums the first `n` integers:

    {mo.ui.code_editor(triangle_text, language="javascript", disabled=True)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    mo.md(r"""## The `func` Dialect""")
    return


@app.cell(hide_code=True)
def _():
    swap_text = """\
    func.func @swap(%a : i32, %b : i32) -> (i32, i32) {
      func.return %b, %a : i32, i32
    }"""
    return (swap_text,)


@app.cell(hide_code=True)
def _(mo, swap_text, xmo):
    mo.md(
        fr"""
        The [func dialect](https://mlir.llvm.org/docs/Dialects/Func/) contains building blocks to model functions and function calls.

        {xmo.module_html(swap_text)}

        The above function takes two 32-bit integers, and returns them in the opposite order
        In this snippet, there are two operations, `func.func` for function definition and `func.return` to specify the returned values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Exercise 1: Your First Function""")
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
def _(exercise_text, first_text_area, mo):
    _first_info_text = exercise_text(first_text_area.value, "first", ((1, 2), (3, 4)), ("first(1, 2) = ", "first(3, 4) = "))
    mo.vstack((first_text_area, mo.md(_first_info_text)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The Arith Dialect""")
    return


@app.cell(hide_code=True)
def _():
    l1_dist_text = """\
    func.func @l1_dist(%a : i32, %b : i32, %c : i32) -> (i32) {
      %a_minus_b = arith.subi %a, %b : i32
      %b_minus_a = arith.subi %b, %a : i32
      %slt = arith.cmpi slt, %lhs, %rhs : i1
      %res = arith.select %slt, %a_minus_b, %b_minus_a : i32
      func.return %res : i32
    }"""
    return (l1_dist_text,)


@app.cell(hide_code=True)
def _(l1_dist_text, mo, xmo):
    mo.md(
        fr"""
        The [arith dialect](https://mlir.llvm.org/docs/Dialects/Arith/) contains arithmetic operations on integers, floating-point values, and other numeric constructs.

        {xmo.module_html(l1_dist_text)}

        The `arith.cmpi` operation specifies how it interprets the inputs.
        Importantly, the signedness of the operands is not specified by the types, and rather by the operation itself.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Exercise 2: Fused Multiply-Add""")
    return


@app.cell(hide_code=True)
def _():
    fma_text = """\
    func.func @fused_multiply_add(%a : i32, %b : i32, %c : i32) -> (i32) {
      // Change this to return a * b + c instead
      func.return %a : i32
    }"""
    return (fma_text,)


@app.cell(hide_code=True)
def _(fma_text, mo):
    fma_text_area = mo.ui.code_editor(fma_text, language="javascript")
    return (fma_text_area,)


@app.cell(hide_code=True)
def _(exercise_text, fma_text_area, mo):
    _fma_info_text = exercise_text(fma_text_area.value, "fused_multiply_add", ((1, 2, 3), (4, 5, 6)), ("first(1, 2, 3) = ", "first(4, 5, 6) = "))
    mo.vstack((fma_text_area, mo.md(_fma_info_text)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The `scf` Dialect""")
    return


@app.cell(hide_code=True)
def _():
    l1_dist_scf_text = """\
    func.func @l1_dist(%a : i32, %b : i32) -> (i32) {
      %slt = arith.cmpi slt, %a, %b : i1
      %res = scf.if %slt -> (i32) {
        %b_minus_a = arith.subi %b, %a : i32
        scf.yield %b_minus_a : i32
      } else {
        %a_minus_b = arith.subi %a, %b : i32
        scf.yield %a_minus_b : i32
      }
      func.return %res : i32
    }"""
    return (l1_dist_scf_text,)


@app.cell(hide_code=True)
def _(l1_dist_scf_text, mo, xmo):
    mo.md(
        fr"""
        The [`scf` dialect](https://mlir.llvm.org/docs/Dialects/Scf/) contains operations for control flow.
        Here is another implementation of l1_distance using an if statement:

        {xmo.module_html(l1_dist_scf_text)}

        Note that we did not put early returns in the branches of the if operation.
        This is because MLIR's SSA blocks have a contract, which is that all the operations are executed from top to bottom, and operations are guaranteed to yield to the outer block.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, triangle_text, xmo):
    mo.md(
        fr"""
        The `scf` dialect also contains abstractions to represent for loops, allowing us to implement the triangle function 1 + 2 + 3 + ... + n.

        {xmo.module_html(triangle_text)}

        Due to the SSA contract, we cannot accumulate by updating a value.
        Instead, the loop body takes in some number of immutable values and yields the same number of values to use for the next loop.
        After all the iterations, these values are returned by the operation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Exercise 3: Factorial Function""")
    return


@app.cell(hide_code=True)
def _(mo, triangle_text):
    second_input_text = mo.ui.text("5")
    second_text_area = mo.ui.code_editor(
        triangle_text.replace("triangle", "factorial"), language="javascript"
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
        second_results = run_func(second_module, "factorial", second_inputs)
        second_results_text = f"factorial({second_input}) = {second_results}"
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
            mo.md(f"""Input: {second_input_text} {mo.ui.button(label="run")}"""),
            second_text_area,
            mo.md(second_info_text)
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The `builtin` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The [`builtin` dialect](https://mlir.llvm.org/docs/Dialects/Builtin/) contains the most commonly-used operation in MLIR/xDSL: `builtin.module`

        A module is a unit of code in xDSL and MLIR.
        It holds a single region.

        The smallest possible piece of code in MLIR IR is an empty module:

        ```
        builtin.module {
        }
        ```

        When the first operation in a file is not a `builtin.module`, it is assumed, as is the case for all the snippets above.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Attributes""")
    return


@app.cell(hide_code=True)
def _(builtin, mo):
    mo.md(fr"""
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
        "some_string": builtin.StringAttr("my_string"),
        "some_int": builtin.IntegerAttr(42, builtin.IndexType()),
        "some_float": builtin.FloatAttr(3.1415, builtin.f32),
        "a_unit_attr": builtin.UnitAttr()
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

    mo.md(fr"""
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
    mo.md(r"""## Generic Format""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The IRs above are in what's called the _custom format_, a format that allows functions to specify a pretty and concise representation.
        The _generic format_ is a more uniform and verbose representation that unambiguously shows the structure of an operation.
        Here is the above minimal function in generic format:
        """
    )
    return


@app.cell(hide_code=True)
def _(Parser, Printer, StringIO, ctx, mo, swap_text):
    _swap_module = Parser(ctx, swap_text).parse_module()
    _file = StringIO()
    Printer(print_generic_format=True, stream=_file).print(_swap_module)
    mo.md(f"""
    ```
    {_file.getvalue()}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(ModuleOp, Printer, StringIO):
    def print_generic(module: ModuleOp) -> str:
        io = StringIO()
        Printer(io, print_generic_format=True).print(module)
        return io.getvalue()
    return (print_generic,)


@app.cell(hide_code=True)
def _(MLContext, arith, builtin, func, scf):
    ctx = MLContext()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(arith.Arith)
    return (ctx,)


@app.cell(hide_code=True)
def _():
    triangle_text = """\
    func.func @triangle(%n: index) -> index {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %n_plus_one = arith.addi %n, %one : index
      %res = scf.for %i = %one to %n_plus_one step %one iter_args(%acc_in = %zero) -> (index) {
        %acc_out = arith.addi %acc_in, %i : index
        scf.yield %acc_out : index
      }
      func.return %res : index
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
def _(Any, ModuleOp):
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
    return (run_func,)


@app.cell(hide_code=True)
def _(Any, Parser, ctx, run_func):
    def exercise_text(module_text: str, function_name: str, inputs: tuple[Any, ...], formats: tuple[str, ...]) -> str:
        error_text = ""
        results_text = ""
        try:
            module = Parser(ctx, module_text).parse_module()
            results_text = "\n".join(
                format + str(run_func(module, function_name, input))
                for format, input in zip(formats, inputs, strict=True)
            )
        except Exception as e:
            error_text = str(e)
        if error_text:
            info_text = f"""
    Error:

    ```
    {error_text}
    ```
    """
        else:
            info_text = f"""
    Here are the outputs when running the function:

    ```
    {results_text}
    ```
    """
        return info_text
    return (exercise_text,)


if __name__ == "__main__":
    app.run()
