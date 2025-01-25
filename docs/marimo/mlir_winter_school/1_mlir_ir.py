import marimo

__generated_with = "0.10.14"
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
    mo.md(fr"""
    MLIR and xDSL use an encoding of the IR as a textual format for debugging, testing, and storing intermediate representations of programs.
    It can be very useful to take a program at some stage of compilation, and inspect it.
    The textual format makes this easy to do.
    Let's look at a representation of a function that sums the first `n` integers:

    ```
    {triangle_text}
    ```

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
    mo.md(fr"""
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
    mo.md(fr"""
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


@app.cell
def _(builtin, mo):
    _module_op = builtin.ModuleOp([])
    _module_op.attributes = {"my_key": builtin.StringAttr("my_value")}

    mo.md(fr"""
    Operations can be supplemented with arbitrary information via their attribute dictionary.

    Here's a module with some extra information:
    ```
    {str(_module_op)}
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.ui.text_area()
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
        %res = scf.for %i = %zero to %n step %step iter_args(%acc_in = %step) -> (index) {
          %square = arith.muli %i, %i : index
          %acc_out = arith.addi %acc_in, %square : index
          scf.yield %acc_out : index
        }
        func.return %res : index
      }
    }
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


if __name__ == "__main__":
    app.run()
