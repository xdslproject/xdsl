import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from xdsl.utils import marimo as xmo
    return (xmo,)


@app.cell(hide_code=True)
def _(mo, xmo):
    from typing import Any
    from io import StringIO

    from xdsl.frontend.listlang.main import ParseError, parse_program
    from xdsl.dialects import builtin
    from xdsl.builder import Builder, InsertPoint
    from xdsl.passes import PassPipeline
    from xdsl.printer import Printer
    from xdsl.transforms import get_all_passes
    from xdsl.context import Context
    from xdsl.frontend.listlang.lowerings import LowerListToTensor

    def to_mlir(code: str) -> builtin.ModuleOp:
        module = builtin.ModuleOp([])
        builder = Builder(InsertPoint.at_start(module.body.block))
        parse_program(code, builder)
        return module

    def compilation_output(code_editor: Any) -> mo.md:
        try:
            return xmo.module_md(to_mlir(code_editor.value))
        except ParseError as e:
            return mo.md(f"Compilation error: {e}")

    def get_compilation_outputs_with_passes(code_editor: Any, pass_editor: Any) -> list[tuple[str, mo.md]] | ParseError:
        try:
            module = to_mlir(code_editor.value)
            module_list = [module.clone()]

            def callback(pass1, module, pass2):
                module_list.append(module.clone())
            all_passes = get_all_passes()
            all_passes["lower-list-to-tensor"] = lambda: LowerListToTensor()
            pipeline = PassPipeline.parse_spec(all_passes, pass_editor.value, callback)
            labels = ["IR before a pass was executed"] + ["IR after " + p.name for p in pipeline.passes]
            pipeline.apply(Context(), module)
            module_list.append(module.clone())
            return [(label, xmo.module_md(module)) for label, module in zip(labels, module_list)]
        except ParseError as e:
            return e
    return compilation_output, get_compilation_outputs_with_passes, to_mlir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # An Introduction to SSA & MLIR

    We explore the ideas behind SSA & MLIR through a small Rust-like Array DSL.
    """
    )
    return


@app.cell(hide_code=True)
def _(check, mo):
    mo.md(f"""<br>\n## Interactive & Reactive! &nbsp; &nbsp;{check}\n\nThis notebook is *reactive*, meaning you can *interact* with our examples. Try the sliders!""")
    return


@app.cell(hide_code=True)
def _(mo):
    interact_x = mo.ui.slider(start=1, stop=30, label="x", value=13)
    interact_y = mo.ui.slider(start=1, stop=30, label="y", value=26)
    mo.hstack([interact_x, interact_y], justify="start")
    return interact_x, interact_y


@app.cell(hide_code=True)
def _(interact_x, interact_y, mo):
    interactive_example_add = "### Addition `+`\n" + ("`" * 3) + "rust\n" + "let x = " + str(interact_x.value) + ";\n" + "let y = " + str(interact_y.value) + ";\n" + "x + y" + "\n" + ("`" * 3)
    code_add = mo.md(interactive_example_add)
    result_add = interact_x.value + interact_y.value
    slider_add = mo.ui.slider(start=1, stop=1000, label="x + y", value=result_add, disabled=True)
    hstack_add = mo.hstack([slider_add, mo.md(str(result_add))], justify="start")
    stack_add = mo.vstack([code_add, hstack_add])

    interactive_example_mul = "### Multiplication `*`\n" + ("`" * 3) + "rust\n" + "let x = " + str(interact_x.value) + ";\n" + "let y = " + str(interact_y.value) + ";\n" + "x * y" + "\n" + ("`" * 3)
    code_mul = mo.md(interactive_example_mul)
    result_mul = interact_x.value * interact_y.value
    slider_mul = mo.ui.slider(start=1, stop=1000, label="x * y", value=result_mul, disabled=True)
    hstack_mul = mo.hstack([slider_mul, mo.md(str(result_mul))], justify="start")
    stack_mul = mo.vstack([code_mul, hstack_mul])

    code_examples = mo.hstack([stack_add, stack_mul])

    check = "✅ " if 10 * result_add == result_mul else "❌" 

    challenge = mo.md("<br>\n### Exercise\nAdjust the sliders such that: `10 * (x + y) = x * y`" +
                     f", &nbsp;&nbsp;&nbsp; {10*result_add} = {result_mul} &nbsp;&nbsp; {check}")

    mo.vstack([code_examples, challenge])
    return (check,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(exp_check, mo):
    mo.md(
        r"""
    <br>
    ## Arithmetic Expressions &nbsp;&nbsp;""" + exp_check + r"""

    Let's explore how our language treats arithmetic expressions.

    ```rust
    let x = 3;
    let y = 5;
    let z = 7;
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    prefix = r"""
       let x = 3;
       let y = 5;
       let z = 7;
    """

    example1 = "x * y + z"

    editor_add_expr = mo.ui.code_editor(value = example1, max_height=1)
    editor_add_expr
    return editor_add_expr, prefix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our compiler translates the code above into MLIR's intermediate representation (IR):""")
    return


@app.cell(hide_code=True)
def _(editor_add_expr, mo, prefix, to_mlir, xmo):
    def num_there(s):
        return any(i.isdigit() for i in s)

    exp_val = editor_add_expr.value
    if exp_val.count("let"):
        res = mo.md(r"""
    /// attention | Error!

    'let' expressions are not allowed in this exercise.
    ///
        """)
        exp_val = ""

    elif num_there(exp_val):
        res = mo.md(r"""
    /// attention | Error!

    Constants are not allowed in this exercise.
    ///
        """)
        exp_val = ""

    else:
        # TODO: Instead of showing the parsing error, can we show the last output
        # plus the error message?
        arithmetic_module = to_mlir(prefix + exp_val)
        res = xmo.module_md(arithmetic_module)

    res
    return (arithmetic_module,)


@app.cell
def _(arithmetic_module):
    print(arithmetic_module)
    return


@app.cell
def _():
    from xdsl.frontend.listlang import marimo as lmo
    return (lmo,)


@app.cell
def _(arithmetic_module, lmo, mo):
    exp_output = lmo.interp(arithmetic_module)
    exp_check = "✅ " if exp_output == "38" else "❌" 
    mo.md(f"Interpreting the IR yields: {exp_output}\n### Exercise\nChange the expression to compute 38. &nbsp;&nbsp; {exp_check}")
    return (exp_check,)


@app.cell
def _(mo):
    mo.md(
        r"""
    /// details | MLIR IR - What do we see?

    1) An MLIR program consists of a list of *operations* (e.g., `arith.constant`, `arith.addi`, `arith.muli`).<br>
    2) The *result* of each operation is assigned to a variable.<br>
    3) Each variable name begins with **%**.<br>
    4) Some operations (e.g., `printf.print_format`) do not yield results and do not define new variables.

    ### The Components of an MLIR Operation

    <span data-tooltip="The return value of the operation">`%c1`</span> = <span data-tooltip="The dialect (namespace) of the operation">`arith`</span>`.`<span data-tooltip="The name of the operation">`constant`</span> <span data-tooltip="Call-site specific static information">1</span> `:` <span data-tooltip="The type of the return value"> !i32</span><br>
    <span data-tooltip="The return value of the operation">`%result`</span> = <span data-tooltip="The dialect (namespace) of the operation">`arith`</span>`.`<span data-tooltip="The name of the operation">`addi`</span> <span data-tooltip="A list of operands">`%c1`, `%c1`</span> `:` <span data-tooltip="The type of the operands and return values"> !i32</span><br>
    <span data-tooltip="The dialect (namespace) of the operation">`printf`</span>`.`<span data-tooltip="The name of the operation">`print_format`</span> <span data-tooltip="Call-site specific static information">`"{}"`</span>`,`  <span data-tooltip="A list of operands">`%result`</span>  `:` <span data-tooltip="The type of the operand"> !i32</span>

    Explore by hovering over the IR.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(bool_all_check, mo):
    mo.md(
         "<br>\n## Boolean Expressions &nbsp;&nbsp;" + bool_all_check + r"""

    Find a Boolean expression that holds for all cases below. Use `true`, `false`, `&&`, `||`, `==`, `!=`, `<`, `>`, `<=`, `>=`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _initial_code = r"""x < y"""

    bool_edit = mo.ui.code_editor(language="rust", value=_initial_code)
    bool_edit
    return (bool_edit,)


@app.cell(hide_code=True)
def _(bool_edit, lmo, mo, to_mlir, xmo):
    bool_1_prefix = r"let x = 10; let y = 12;"

    bool_1_module = to_mlir(bool_1_prefix + bool_edit.value)
    bool_1_output = lmo.interp(bool_1_module)
    bool_1_expected = "true"
    bool_1_ok = bool_1_output == bool_1_expected
    bool_1_check = "✅ " if bool_1_ok else "❌"

    bool_1_cmp = mo.md(f"expected: {bool_1_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_1_output}")
    bool_1_stack = mo.vstack([mo.md("### Case 1"), lmo.rust_md(bool_1_prefix),  bool_1_cmp, mo.md(bool_1_check)])

    bool_2_prefix = "let x = 13; let y = 18;"

    bool_2_module = to_mlir(bool_2_prefix + bool_edit.value)
    bool_2_output = lmo.interp(bool_2_module)
    bool_2_expected = "false"
    bool_2_ok = bool_2_output == bool_2_expected
    bool_2_check = "✅ " if bool_2_ok else "❌"

    bool_2_cmp = mo.md(f"expected: {bool_2_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_2_output}")
    bool_2_stack = mo.vstack([mo.md("### Case 2"), lmo.rust_md(bool_2_prefix), bool_2_cmp, mo.md(bool_2_check)])

    bool_res = xmo.module_md(to_mlir(bool_1_prefix + bool_edit.value))

    bool_3_prefix = r"let x = 8; let y = 2;"

    bool_3_module = to_mlir(bool_3_prefix + bool_edit.value)
    bool_3_output = lmo.interp(bool_3_module)
    bool_3_expected = "true"
    bool_3_ok = bool_3_output == bool_3_expected
    bool_3_check = "✅ " if bool_3_ok else "❌" 

    bool_3_cmp = mo.md(f"expected: {bool_3_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_1_output}")
    bool_3_stack = mo.vstack([mo.md("### Case 3"), lmo.rust_md(bool_3_prefix),  bool_3_cmp, mo.md(bool_3_check)])

    bool_4_prefix = "let x = 27; let y = 18;"

    bool_4_module = to_mlir(bool_4_prefix + bool_edit.value)
    bool_4_output = lmo.interp(bool_4_module)
    bool_4_expected = "false"
    bool_4_ok = bool_4_output == bool_4_expected
    bool_4_check = "✅ " if bool_4_ok else "❌" 

    bool_4_cmp = mo.md(f"expected: {bool_4_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_4_output}")
    bool_4_stack = mo.vstack([mo.md("### Case 4"), lmo.rust_md(bool_4_prefix), bool_4_cmp, mo.md(bool_4_check)])

    bool_res = xmo.module_md(to_mlir(bool_1_prefix + bool_edit.value))


    bool_all_ok = bool_1_ok and bool_2_ok and bool_3_ok and bool_4_ok
    bool_all_check = "✅ " if bool_all_ok else "❌" 

    mo.vstack([bool_res, mo.hstack([bool_1_stack, bool_2_stack]), mo.md("<br>"), 
    mo.hstack([bool_3_stack, bool_4_stack])])
    return (bool_all_check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | What did you notice?

    - Booleans are represented as `i1` (a single bit integer)
    - All comparison operations are
        - encoded as `arith.cmpi` operations
        - have an opcode (`eq`, `ult`, `ule`, ...)
        - have two operands of a potentially wider integer type, e.g., `i32`
        - and return their boolean result as `i1`
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    ## Static Single-Assignment (SSA)

    MLIR IR uses **single static-assignment form** (SSA). In short, this means that every value (variable) is defined only once, and temporary values are defined for each intermediate expressions. In particular, this means that shadowed variables are redefined with a new name. In this notebook, temporary variables are preceeded by an underscore, and shadowed variables have an integer append to their name.

    As an exercise, try to modify the following program so that the generated MLIR code contains no new intermediate values, and no new values introduced by shadowing.
    """
    )
    return


@app.cell
def _(mo):
    reset_button3 = mo.ui.button(label="reset")
    reset_button3
    return (reset_button3,)


@app.cell(hide_code=True)
def _(mo, reset_button3):
    reset_button3

    _initial_code = r"""let x = 3 + 10 + 7;
    let x = x + 1;
    x * 2"""

    example_editor3 = mo.ui.code_editor(language="rust", value=_initial_code)
    example_editor3
    return (example_editor3,)


@app.cell(hide_code=True)
def _(compilation_output, example_editor3):
    compilation_output(example_editor3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Applying compilation passes

    Now that we have showed some MLIR IR code, let's see how we can manipulate MLIR IR with passes. Here are two very important MLIR passes that are relevant in most compilers:

    * `cse` (Common sub-expression elimination): This pass finds operations that are identical, and replace one with the other to reduce the number of computations.
    * `canonicalize`: This pass does three different optimizations in one:
        * It removes operations without side-effects that are not used.
        * It constant fold operations, meaning that operations with only constant inputs will be replaced by a constant.
        * It applies simple local optimizations, such as `x - x = 0`.


    In this notebook, we will define compiler pipelines using a comma-separated list of pass names (for instance, `cse,canonicalize`). The following two code editors will allow you to write a program, a pass pipeline, and see the resulting MLIR IR at each step of the pipeline.
    """
    )
    return


@app.cell
def _(mo):
    reset_button4 = mo.ui.button(label="reset")
    reset_button4
    return (reset_button4,)


@app.cell(hide_code=True)
def _(mo, reset_button4):
    reset_button4

    _initial_code = r"""let a = true;
    let b = false;
    a && b"""

    example_editor4 = mo.ui.code_editor(language="rust", value=_initial_code, label="MLIR code:")
    pass_editor4 = mo.ui.code_editor(value="dce,cse,canonicalize", max_height=1, label="Passes:")

    mo.vstack([example_editor4, pass_editor4])
    return example_editor4, pass_editor4


@app.cell(hide_code=True)
def _(example_editor4, get_compilation_outputs_with_passes, pass_editor4):
    outputs4 = get_compilation_outputs_with_passes(example_editor4, pass_editor4)
    return (outputs4,)


@app.cell
def _(mo, pass_editor4):
    slider4 = mo.ui.slider(start=0, stop=len(pass_editor4.value.split(",")))
    slider4
    return (slider4,)


@app.cell
def _(mo, outputs4, slider4):
    mo.vstack([*outputs4[slider4.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Control flow with Regions

    Now that we have shown how computations work, let's look at how to model control-flow.

    ### `scf.if`

    As a simple first example of control-flow, let's look at a ternary condition. The following example computes the minimum between `x` and `y`:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    example5 = r"""
    let x = 5;
    let y = 7;
    if x < y {x} else {y}
    """

    mo.ui.code_editor(language = "rust", value = example5, disabled = True)
    return (example5,)


@app.cell
def _(example5, to_mlir, xmo):
    xmo.module_md(to_mlir(example5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can see the following in the generated MLIR IR:

    * The `if` is represented using an `scf.if` operation. `scf` stands for "structured control-flow".
    * `scf.if` contains two **regions**. A region is a section of code that contains another list of operation (we will explain this more in details later).
    * Only one region is executed, this logic comes from `scf.if`. In general, operations that have regions decide when they are executed through compilation passes.
    * Each region ends with an `scf.yield`, this is the "value" that is leaving the region when the region gets executed, and that gets returned by the `scf.if` operation.

    Try to change the following program, and look at the effect of different optimization passes!
    """
    )
    return


@app.cell
def _(mo):
    reset_button6 = mo.ui.button(label="reset")
    reset_button6
    return (reset_button6,)


@app.cell(hide_code=True)
def _(mo, reset_button6):
    reset_button6

    _initial_code = r"""let x = 5;
    let y = 7;
    if x < y {x} else {y}"""

    example_editor6 = mo.ui.code_editor(language="rust", value=_initial_code, label="MLIR code:")
    pass_editor6 = mo.ui.code_editor(value="cse,canonicalize", max_height=1, label="Passes:")

    mo.vstack([example_editor6, pass_editor6])
    return example_editor6, pass_editor6


@app.cell(hide_code=True)
def _(example_editor6, get_compilation_outputs_with_passes, pass_editor6):
    outputs6 = get_compilation_outputs_with_passes(example_editor6, pass_editor6)
    return (outputs6,)


@app.cell(hide_code=True)
def _(mo, pass_editor6):
    slider6 = mo.ui.slider(start=0, stop=len(pass_editor6.value.split(",")))
    slider6
    return (slider6,)


@app.cell(hide_code=True)
def _(mo, outputs6, slider6):
    mo.vstack([*outputs6[slider6.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adding lists to our DSL

    ### Adding abstractions with dialects

    Now that we have seen most of the core concepts of MLIR IR, let's now add support for lists in our DSL.
    Our list DSL has the following operations:

    * Creating a list from a range (`x..y`)
    * Getting the length of a list (`list.len()`)
    * Mapping a function over a list (`list.map(|x| x + 1)`)

    In order to represent these lists and operations in MLIR IR, we will create our custom new operations and types.
    This is done through defining what's called a **dialect**, a namespace for a set of operations and types. One of the main advantages of defining a custom dialect for our lists, is that we can now define simple optimizations that are specific to our dialect, and that would be hard to do otherwise. For instance, we can extend the `canonicalize` pass to understand how to optimize `list.map` operations.

    Here is an example of a program using lists, feel free to modify it and see the generated MLIR code:
    """
    )
    return


@app.cell
def _(mo):
    reset_button7 = mo.ui.button(label="reset")
    reset_button7
    return (reset_button7,)


@app.cell(hide_code=True)
def _(mo, reset_button7):
    reset_button7

    _initial_code = r"""let a = 0..10;
    let c = a.map(|x| x + a.len());
    c"""

    example_editor7 = mo.ui.code_editor(language="rust", value=_initial_code, label="MLIR code:")
    pass_editor7 = mo.ui.code_editor(value="cse,canonicalize", max_height=1, label="Passes:")

    mo.vstack([example_editor7, pass_editor7])
    return example_editor7, pass_editor7


@app.cell(hide_code=True)
def _(example_editor7, get_compilation_outputs_with_passes, pass_editor7):
    outputs7 = get_compilation_outputs_with_passes(example_editor7, pass_editor7)
    return (outputs7,)


@app.cell
def _(mo, pass_editor7):
    slider7 = mo.ui.slider(start=0, stop=len(pass_editor7.value.split(",")))
    slider7
    return (slider7,)


@app.cell
def _(mo, outputs7, slider7):
    mo.vstack([*outputs7[slider7.value]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note the following:

    * The list type is represented as `!list.list`. This is how custom types are represented in MLIR IR `!dialect.type`.
    * The custom operations, as well as the custom type all start with `list`, which is the name of our dialect.
    * `list.map` uses a region to represent the function to be applied on each element of the list. This region has an argument,
        `x`, which is the element of the list being processed. The `list.yield` operation is used to return the new value for
        the element.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Lowering our abstractions to MLIR

    Once our frontend produces MLIR IR, we can use passes to lower (compile) our `list` dialect to existing MLIR dialect. From there, we can use existing MLIR passes to lower our code to LLVM.

    As we are dealing with arrays, we can compile our code to the `tensor` abstraction, along with the `scf` abstraction for control flow such as loops.

    The tensor type we are using is `tensor<?xi32>`. This represents tensors of rank 1, with an arbitrary dimenson. We can construct an empty tensor with `tensor.empty`, and write in it with `tensor.insert`. Tensors have *value-semantics*, meaning that a `tensor.insert` returns a new tensor, and the previous one can still be reused. The way tensors are layed out in memory is defined by a lowering pass called bufferization.

    In order to write an entire tensor, we use `scf.for`. This is an operation with a single region with two arguments. The region argument is the iterator value, and the second one is the value that is passed to the region and then returned. It is used to represent the accumulation of a value using SSA.
    """
    )
    return


@app.cell
def _(mo):
    reset_button8 = mo.ui.button(label="reset")
    reset_button8
    return (reset_button8,)


@app.cell(hide_code=True)
def _(mo, reset_button8):
    reset_button8

    _initial_code = r"""let a = 0..10;
    let c = a.map(|x| x + a.len());
    c"""

    example_editor8 = mo.ui.code_editor(language="rust", value=_initial_code, label="MLIR code:")
    pass_editor8 = mo.ui.code_editor(value="cse,canonicalize,lower-list-to-tensor,cse,licm,canonicalize", max_height=1, label="Passes:")

    mo.vstack([example_editor8, pass_editor8])
    return example_editor8, pass_editor8


@app.cell
def _(example_editor8, get_compilation_outputs_with_passes, pass_editor8):
    outputs8 = get_compilation_outputs_with_passes(example_editor8, pass_editor8)
    return (outputs8,)


@app.cell
def _(mo, pass_editor8):
    slider8 = mo.ui.slider(start=0, stop=len(pass_editor8.value.split(",")))
    slider8
    return (slider8,)


@app.cell
def _(mo, outputs8, slider8):
    mo.vstack([*outputs8[slider8.value]])
    return


if __name__ == "__main__":
    app.run()
