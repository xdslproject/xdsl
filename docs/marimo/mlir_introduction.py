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


@app.cell
def _(mo, xmo):
    from typing import Any
    from io import StringIO

    from xdsl.frontend.listlang.main import ParseError, parse_program
    from xdsl.dialects import builtin
    from xdsl.dialects import get_all_dialects
    from xdsl.builder import Builder, InsertPoint
    from xdsl.passes import PassPipeline
    from xdsl.printer import Printer
    from xdsl.parser import Parser
    from xdsl.transforms import get_all_passes
    from xdsl.transforms.dead_code_elimination import DeadCodeElimination
    from xdsl.transforms.constant_fold_interp import ConstantFoldInterpPass
    from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
    from xdsl.context import Context
    from xdsl.frontend.listlang.lowerings import LowerListToTensor
    from xdsl.utils.exceptions import VerifyException, InterpretationError
    from xdsl.utils.exceptions import ParseError as XDSLParseError
    from xdsl.frontend.listlang.transforms import OptimizeListOps

    all_dialects = get_all_dialects()

    def to_mlir(code: str) -> builtin.ModuleOp:
        module = builtin.ModuleOp([])
        builder = Builder(InsertPoint.at_start(module.body.block))
        parse_program(code, builder)
        return module

    def parse_mlir(code: str) -> builtin.ModuleOp:
        context = Context()
        for dialect in ("builtin", "scf", "arith", "printf"):
            context.register_dialect(dialect, all_dialects[dialect])
        parser = Parser(context, code)
        return parser.parse_module(code)

    def compilation_output(code_editor: Any) -> mo.md:
        try:
            return xmo.module_md(to_mlir(code_editor.value))
        except ParseError as e:
            return mo.md(f"Compilation error: {e}")

    def get_compilation_outputs_with_passes(code_editor: Any, pass_editor: Any, input="rs", result="md") -> list[tuple[str, mo.md]]:
        if input == "rs":
            module = to_mlir(code_editor.value)
        else:
            module = parse_mlir(code_editor.value)
        module_list = [module.clone()]
        def callback(pass1, module, pass2):
            module_list.append(module.clone())
        all_passes = get_all_passes()
        all_passes["lower-list-to-tensor"] = lambda: LowerListToTensor()
        all_passes["optimize-lists"] = lambda: OptimizeListOps()
        pipeline = PassPipeline.parse_spec(all_passes, pass_editor.value, callback)
        titles = xmo.pipeline_titles(pipeline.passes)
        labels = ["Initial IR"] + ["IR after " + t for t in titles]
        pipeline.apply(Context(), module)
        module_list.append(module.clone())
        if result == "md":
            return [(label, xmo.module_md(module)) for label, module in zip(labels, module_list, strict=True)]
        return [(label, module) for label, module in zip(labels, module_list, strict=True)]

    def execute_and_catch_exceptions(fun: Any) -> Any | tuple[bool, mo.md]:
        """
        execute a lambda, and return a formatted error if any exception happened.
        """
        try:
            return fun()
        except ParseError as e:
            _error_output9 = StringIO()
            print(e, file=_error_output9)
            return False, mo.md("/// attention | Compilation error:\n" + "`" * 3 + "\n" + _error_output9.getvalue() + "`" * 3 + "\n///")
        except XDSLParseError as e:
            _error_output9 = StringIO()
            print(e, file=_error_output9)
            return False, mo.md("/// attention | Compilation error:\n" + "`" * 3 + "\n" + _error_output9.getvalue() + "`" * 3 + "\n///")
        except InterpretationError as e:
            _error_output9 = StringIO()
            print(e, file=_error_output9)
            return False, mo.md("/// attention | Interpretation error:\n" + "`" * 3 + "\n" + _error_output9.getvalue() + "`" * 3 + "\n///")
        except VerifyException as e:
            _error_output9 = StringIO()
            print(e.__notes__[0], file=_error_output9)
            print(e, file=_error_output9)
            return False, mo.md("/// attention | Compilation error:\n" + "`" * 3 + "\n" + _error_output9.getvalue() + "`" * 3 + "\n///")
    return (
        CommonSubexpressionElimination,
        ConstantFoldInterpPass,
        Context,
        DeadCodeElimination,
        LowerListToTensor,
        OptimizeListOps,
        PassPipeline,
        compilation_output,
        execute_and_catch_exceptions,
        get_all_passes,
        get_compilation_outputs_with_passes,
        parse_mlir,
        to_mlir,
    )


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
def _(mo):
    mo.md(
        f"""
    <br>
    ## Interactive & Reactive!

    This notebook is *reactive*, meaning you can *interact* with our examples. Try the sliders!
    """
    )
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

    challenge = mo.md(f"<br>\n### Exercise &nbsp; &nbsp;{check}\nAdjust the sliders such that: `10 * (x + y) = x * y`" +
                     f", &nbsp;&nbsp;&nbsp; {10*result_add} = {result_mul} &nbsp;&nbsp; {check}")

    mo.vstack([code_examples, challenge])
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    ## Arithmetic Expressions""" + r"""

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

    editor_add_expr = mo.ui.code_editor(language="rs", value = example1, max_height=1)
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
        _res = mo.md(r"""
    /// attention | Error!

    'let' expressions are not allowed in this exercise.
    ///
        """)
        exp_val = ""

    elif num_there(exp_val):
        _res = mo.md(r"""
    /// attention | Error!

    Constants are not allowed in this exercise.
    ///
        """)
        exp_val = ""

    else:
        # TODO: Instead of showing the parsing error, can we show the last output
        # plus the error message?
        arithmetic_module = to_mlir(prefix + exp_val)
        _res = xmo.module_md(arithmetic_module)

    _res
    return (arithmetic_module,)


@app.cell
def _():
    from xdsl.frontend.listlang import marimo as lmo
    return (lmo,)


@app.cell
def _(arithmetic_module, lmo, mo):
    exp_output = lmo.interp(arithmetic_module)
    exp_check = "✅ " if exp_output == "38" else "❌"
    mo.md(f"Interpreting the IR yields: {exp_output}\n### Exercise &nbsp;&nbsp; {exp_check} \nChange the expression to compute 38. &nbsp;&nbsp; {exp_check}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    /// details | MLIR IR - What do we see?

    1) An MLIR program consists of a list of *operations* (e.g., `arith.constant`, `arith.addi`, `arith.muli`).<br>
    2) The *result* of each operation is assigned to a variable.<br>
    3) Each variable name begins with **%**.<br>
    4) Some operations (e.g., `printf.print_format`) do not yield results and do not define new variables.

    ///

    ### The Components of an MLIR Operation

    <span data-tooltip="The return value of the operation">`%c1`</span> = <span data-tooltip="The dialect (namespace) of the operation">`arith`</span>`.`<span data-tooltip="The name of the operation">`constant`</span> <span data-tooltip="Call-site specific static information">1</span> `:` <span data-tooltip="The type of the return value"> i32</span><br>
    <span data-tooltip="The return value of the operation">`%result`</span> = <span data-tooltip="The dialect (namespace) of the operation">`arith`</span>`.`<span data-tooltip="The name of the operation">`addi`</span> <span data-tooltip="A list of operands">`%c1`, `%c1`</span> `:` <span data-tooltip="The type of the operands and return values"> i32</span><br>
    <span data-tooltip="The dialect (namespace) of the operation">`printf`</span>`.`<span data-tooltip="The name of the operation">`print_format`</span> <span data-tooltip="Call-site specific static information">`"{}"`</span>`,`  <span data-tooltip="A list of operands">`%result`</span>  `:` <span data-tooltip="The type of the operand"> i32</span>

    Explore by hovering over the IR.
    """
    )
    return


@app.cell(hide_code=True)
def _(bool_all_check, mo):
    mo.md(
        f"""
    <br>\n## Boolean Expressions

    ### Exercise {bool_all_check}

    Find a Boolean expression that holds for all cases below. Use `true`, `false`, `&&`, `||`, `==`, `!=`, `<`, `>`, `<=`, `>=`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _initial_code = r"""x < y"""

    bool_edit = mo.ui.code_editor(language="rs", value=_initial_code)
    bool_edit
    return (bool_edit,)


@app.cell(hide_code=True)
def _(bool_edit, lmo, mo, to_mlir, xmo):
    bool_1_prefix = r"let x = 5; let y = 9;"

    bool_1_module = to_mlir(bool_1_prefix + bool_edit.value)
    bool_1_output = lmo.interp(bool_1_module)
    bool_1_expected = "true"
    bool_1_ok = bool_1_output == bool_1_expected
    bool_1_check = "✅ " if bool_1_ok else "❌"

    bool_1_cmp = mo.md(f"expected: {bool_1_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_1_output}")
    bool_1_stack = mo.vstack([mo.md("### Case 1 &nbsp;&nbsp;" + bool_1_check), lmo.rust_md(bool_1_prefix), bool_1_cmp])

    bool_2_prefix = "let x = 7; let y = 7;"

    bool_2_module = to_mlir(bool_2_prefix + bool_edit.value)
    bool_2_output = lmo.interp(bool_2_module)
    bool_2_expected = "false"
    bool_2_ok = bool_2_output == bool_2_expected
    bool_2_check = "✅ " if bool_2_ok else "❌"

    bool_2_cmp = mo.md(f"expected: {bool_2_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_2_output}")
    bool_2_stack = mo.vstack([mo.md("### Case 2 &nbsp;&nbsp;" + bool_2_check), lmo.rust_md(bool_2_prefix), bool_2_cmp])

    bool_res = xmo.module_md(to_mlir(bool_1_prefix + bool_edit.value))

    bool_3_prefix = r"let x = 8; let y = 2;"

    bool_3_module = to_mlir(bool_3_prefix + bool_edit.value)
    bool_3_output = lmo.interp(bool_3_module)
    bool_3_expected = "true"
    bool_3_ok = bool_3_output == bool_3_expected
    bool_3_check = "✅ " if bool_3_ok else "❌"

    bool_3_cmp = mo.md(f"expected: {bool_3_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_1_output}")
    bool_3_stack = mo.vstack([mo.md("### Case 3 &nbsp;&nbsp;" + bool_3_check), lmo.rust_md(bool_3_prefix), bool_3_cmp])

    bool_4_prefix = "let x = 3; let y = 5;"

    bool_4_module = to_mlir(bool_4_prefix + bool_edit.value)
    bool_4_output = lmo.interp(bool_4_module)
    bool_4_expected = "true"
    bool_4_ok = bool_4_output == bool_4_expected
    bool_4_check = "✅ " if bool_4_ok else "❌"

    bool_4_cmp = mo.md(f"expected: {bool_4_expected}" + "&nbsp; &nbsp; ↔ &nbsp; " + f"current: {bool_4_output}")
    bool_4_stack = mo.vstack([mo.md("### Case 4 &nbsp;&nbsp;" + bool_4_check), lmo.rust_md(bool_4_prefix), bool_4_cmp])

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


@app.cell
def _(mo):
    mo.md(r"""## Get your hands dirty - with the `arith` dialect""")
    return


@app.cell
def _(match_check, mo):
    mo.md("### Exercise: Match an MLIR Program &nbsp;&nbsp;" + match_check)
    return


@app.cell
def _(mo, to_mlir, xmo):
    match_listlang = r"""
    let c = 100;
    let x = 42;
    x + c
    """

    match_instruction = mo.md("Write a small program that yields the following MLIR output:")

    match_editor = mo.ui.code_editor(value = "", max_height=1, placeholder="let ...")
    mo.vstack([match_instruction, xmo.module_md(to_mlir(match_listlang)), match_editor])
    return match_editor, match_listlang


@app.cell
def _(match_editor, match_listlang, mo, to_mlir, xmo):
    match_mlir = to_mlir(match_editor.value)
    match_mlir_md = xmo.module_md(match_mlir)

    match_check = "✅ " if str(match_mlir) == str(to_mlir(match_listlang)) else "❌"

    mo.vstack([match_mlir_md, match_check])
    return (match_check,)


@app.cell
def _(mo, write_check):
    mo.md(rf"""### Exercise: Write your own MLIR program &nbsp;&nbsp; {write_check}""")
    return


@app.cell
def _(lmo, mo):
    write_listlang = r"""
    let c = 57;
    let x = 47;
    x * c
    """

    write_instruction = mo.md("Write MLIR syntax that matches the program above:")

    write_editor = mo.ui.code_editor(value = "", max_height=1, placeholder="%c = ...")
    mo.vstack([write_instruction, lmo.rust_md(write_listlang), write_editor])
    return write_editor, write_listlang


@app.cell
def _(mo, to_mlir, write_editor, write_listlang):
    write_mlir = to_mlir(write_listlang)

    write_check = "✅ " if str(write_mlir) == str(write_editor.value) else "❌"

    write_hint = """/// details | Need a hint?
    * Make sure the names are correct
    * Did you add the print statement at the end?
    ///"""

    mo.vstack([mo.md(write_check), mo.md(write_hint)])
    return (write_check,)


@app.cell(hide_code=True)
def _(check_ssa, mo):
    mo.md(
        rf"""
    <br>
    ## Static Single-Assignment (SSA)

    MLIR IR uses **single static-assignment form** (SSA). In short, this means that every value (variable) is defined only once, and temporary values are defined for each intermediate expressions. We add an `_` on each variable name introduced to satisfy SSA.

    ### Exercise &nbsp;&nbsp; {check_ssa}

    Try to SSA-ify the following rust program to make it look like the result on the right!
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

    example_editor3 = mo.ui.code_editor(language="rs", value=_initial_code)
    example_editor3
    return (example_editor3,)


@app.cell(hide_code=True)
def _(compilation_output, example_editor3, mo, to_mlir, xmo):
    _result_rust3 = r"""
    let c3 = 3;
    let c10 = 10;
    let t13 = c3 + c10;
    let c7 = 7;
    let x1 = t13 + c7;
    let c1 = 1;
    let x2 = x1 + c1;
    let c2 = 2;
    let res = x2 * c2;
    res
    """

    _user_output = compilation_output(example_editor3)
    _result_output = xmo.module_md(to_mlir(_result_rust3))
    check_ssa = "✅ " if _user_output.text == _result_output.text else "❌"
    _result_md = mo.md("✅ Correct!" if _user_output.text == _result_output.text else "❌ The two outputs are different")
    mo.vstack([mo.hstack([_user_output, _result_output], justify="space-between"), _result_md], align="center")
    return (check_ssa,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    ## Control flow with Regions

    Now, let's look at how MLIR can handle control-flow with **regions**!

    Here is a simple program that has an if condition. Take a look at how it is compiled to MLIR, and feel free to change the original program. What happens when you nest if statements?
    """
    )
    return


@app.cell
def _(mo):
    reset_button5 = mo.ui.button(label="reset")
    reset_button5
    return (reset_button5,)


@app.cell(hide_code=True)
def _(mo, reset_button5):
    reset_button5

    example5 = r"""let x = 5;
    let y = 2;
    if x < y {y + x} else {y}"""
    editor5 = mo.ui.code_editor(language = "rs", value = example5, disabled = False)
    editor5
    return (editor5,)


@app.cell
def _(compilation_output, editor5):
    compilation_output(editor5)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    /// details | What do you see?
    * If expressions are represented by an `scf.if`.
    * An `scf.if` contains two regions, similar to an if expression.
    * Each region ends with an operation called a **terminator**. For `scf.if`, it is an `scf.yield`, and it returns the value computed in the region.
    ///
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Here is the skeleton of an `scf.if`:

    <span data-tooltip="The if expression return value">`%res`</span> = <span data-tooltip="The operation name">scf.if</span> <span data-tooltip="The condition (of type i1)"> `%cond` </span> `->` <span data-tooltip="The if expression return type">`(i32)`</span> {<br>
      &nbsp;&nbsp;&nbsp;&nbsp;<span data-tooltip="The operations to execute in the true branch">`%val_true = arith.addi %x, %y : i32`</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;<span data-tooltip="The scf.if terminator operation">`scf.yield`</span> <span data-tooltip="The value to return when the condition is true">`%val_true`</span> `:` <span data-tooltip="The if expression return type">`i32`</span><br>
    `} else {`<br>
      &nbsp;&nbsp;&nbsp;&nbsp;<span data-tooltip="The operations to execute in the false branch">`%val_false = arith.addi %x, %y : i32`</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;<span data-tooltip="The scf.if terminator operation">`scf.yield`</span> <span data-tooltip="The value to return when the condition is false">`%val_false`</span> `:` <span data-tooltip="The if expression return type">`i32`</span><br>
    }
    """
    )
    return


@app.cell
def _(exercise8_tick, mo):
    mo.md(
        rf"""
    <br>
    ## Get your hands dirty - with the `scf` dialect

    ### Exercise: Minimum of 2 values &nbsp;&nbsp; {exercise8_tick}

    Write the MLIR code that computes the minimum of 2 values.
    Use the variables `%x` and `%y`, and place the result in the `%res` variable. For comparisons, use signed opcodes (e.g. `slt, sle`).
    """
    )
    return


@app.cell
def _(mo):
    editor8 = mo.ui.code_editor(placeholder="%res = ...")

    # Solution:
    """
    %c = arith.cmpi slt, %x, %y : i32
    %res = scf.if %c -> (i32) {
      scf.yield %x: i32
    } else {
      scf.yield %y : i32
    }
    """

    editor8
    return (editor8,)


@app.cell
def _(editor8, lmo, parse_mlir):
    def run8_with_values(x: int, y: int):
        code = (
            f"%x = arith.constant {x} : i32\n"
            f"%y = arith.constant {y} : i32\n"
            + editor8.value + "\n"
            + 'printf.print_format "{}", %res : i32'
        )
        module = parse_mlir(code)
        module.verify()
        result = int(lmo.interp(module))
        return result
    return (run8_with_values,)


@app.cell
def _(editor8, execute_and_catch_exceptions, mo, run8_with_values):
    # Wrap this in a function so we can use early return
    def _execute() -> tuple[bool, mo.md]:
        if "%res" not in editor8.value:
            return False, mo.md("""/// attention | You need to output the result in the %res value!\n///""")
        _test_values = [(4, 4), (4, 8), (5, 4)]
        for (x, y) in _test_values:
            _res = run8_with_values(x, y)
            _expected = min(x, y)
            if _res != min(x, y):
                return False, mo.md(f"❌ Incorrect value for `x = {x}` and `y = {y}`. Expected `{_expected}`, but got `{_res}`")
        return True, mo.md("✅ The program is correct!")

    _correct, _output = execute_and_catch_exceptions(_execute)

    exercise8_tick = "✅" if _correct else "❌"
    _output
    return (exercise8_tick,)


@app.cell
def _(exercise9_tick, mo):
    mo.md(
        rf"""
    ### Exercise: Minimum of 3 values &nbsp;&nbsp; {exercise9_tick}

    Write the MLIR code that computes the minimum of 3 values.
    Use the variables `%x`, `%y`, and `%z`, and place the result in the `%res` variable. For comparisons, use signed opcodes (e.g. `slt, sle`).
    """
    )
    return


@app.cell
def _(mo):
    editor9 = mo.ui.code_editor(placeholder="%res = ...")

    # Solution:
    """
    %c = arith.cmpi slt, %x, %y : i32
    %min1 = scf.if %c -> (i32) {
      scf.yield %x : i32
    } else {
      scf.yield %y : i32
    }
    %c2 = arith.cmpi slt, %min1, %z : i32
    %res = scf.if %c2 -> (i32) {
      scf.yield %min1 : i32
    } else {
      scf.yield %z : i32
    }
    """

    editor9
    return (editor9,)


@app.cell
def _(editor9, lmo, parse_mlir):
    def run9_with_values(x: int, y: int, z: int):
        code = (
            f"%x = arith.constant {x} : i32\n"
            f"%y = arith.constant {y} : i32\n"
            f"%z = arith.constant {z} : i32\n"
            + editor9.value + "\n"
            + 'printf.print_format "{}", %res : i32'
        )
        module = parse_mlir(code)
        module.verify()
        result = int(lmo.interp(module))
        return result
    return (run9_with_values,)


@app.cell
def _(editor9, execute_and_catch_exceptions, mo, run9_with_values):
    # Check that the program compiles and runs correctly
    def _execute():
        if "%res" not in editor9.value:
            return False, mo.md("""/// attention | You need to output the result in the %res value!\n///""")
        _test_values = [(1, 3, 7), (3, 5, 9), (3, 1, 4), (4, 12, 1), (4, 1, 2), (6, 5, 2), (3, 3, 3), (1, 1, 7), (3, 4, 3), (9, 1, 1)]
        for (x, y, z) in _test_values:
            _res = run9_with_values(x, y, z)
            _expected = min(x, y, z)
            if _res != min(x, y, z):
                return False, mo.md(f"❌ Incorrect value for `x = {x}`, `y = {y}`, and `z = {z}`. Expected `{_expected}`, but got `{_res}`")
        return True, mo.md("✅ The program is correct!")

    _correct, _output = execute_and_catch_exceptions(_execute)

    exercise9_tick = "✅" if _correct else "❌"
    _output
    return (exercise9_tick,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    ## Applying compilation passes

    Once we have an MLIR IR program, we can apply a compilation **pass**.

    Here is a list of passes that we will cover in the following sections:

    * `cse` (Constant Sub-expression Elimination): De-duplicate identical operations.
    * `dce` (Dead-Code Elimination): Removes unused side-effect free operations.
    * `constant-fold-interp`: Evaluate operations that only have constant inputs.
    """
    )
    return


@app.cell(hide_code=True)
def _(check_optimizations, mo):
    mo.md(
        rf"""
    ## What passes optimize these programs? &nbsp;&nbsp; {check_optimizations}

    For each of the following programs, what passes do you think will modify the program?
    """
    )
    return


@app.cell
def _(mo):
    get_checkboxes_state, set_checkboxes_state = mo.state(False)

    def build_example(num: int, mlir_str: str) -> tuple[list[mo.ui.checkbox], mo.vstack]:
        title = mo.md(f"#### Example {num}")
        pass_md = mo.md("`" * 3 + "mlir\n" + mlir_str + "`" * 3)
        pass_boxes = [
            mo.ui.checkbox(label=label, on_change=set_checkboxes_state)
            for label in ("cse", "dce", "constant-fold-interp")
        ]
        pass_mo = mo.vstack([title, pass_md, *pass_boxes])
        return (pass_boxes, pass_mo)

    # cse
    pass_1_mlir = r"""%b = arith.addi %a, %a : i32
    %c = arith.addi %a, %a : i32
    %d = arith.addi %b, %c : i32
    printf.print_format "{}", %d : i32
    """
    pass_1_boxes, pass_1_mo = build_example(1, pass_1_mlir)

    # dce
    pass_2_mlir = r"""%b = arith.addi %a, %a : i32
    %c = arith.muli %b, %b : i32
    %d = arith.addi %b, %b : i32
    printf.print_format "{}", %d : i32
    """
    pass_2_boxes, pass_2_mo = build_example(2, pass_2_mlir)

    # nothing
    pass_3_mlir = r"""%t = arith.addi %x, %y : i32
    %t2 = arith.addi %y, %x : i32
    %res = arith.addi %t, %t2 : i32
    printf.print_format "{}", %res : i32
    """
    pass_3_boxes, pass_3_mo = build_example(3, pass_3_mlir)

    # constant-fold-interp
    pass_4_mlir = r"""%x = arith.constant 3 : i32
    %y = arith.constant 15 : i32
    %res = arith.subi %x, %y : i32
    printf.print_format "{}", %res : i32
    """
    pass_4_boxes, pass_4_mo = build_example(4, pass_4_mlir)

    # nothing
    pass_5_mlir = r"""%c1 = arith.constant -1 : i32
    %c0 = arith.constant 0 : i32
    %t = arith.addi %x, %c0 : i32
    %u = arith.muli %x, %c1 : i32
    %res = arith.addi %t, %u : i32
    printf.print_format "{}", %res : i32
    """
    pass_5_boxes, pass_5_mo = build_example(5, pass_5_mlir)

    # dce,constant-fold-interp
    pass_6_mlir = r"""%c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %u = arith.addi %c2, %c4 : i32
    %t = arith.addi %x, %y : i32
    %res = arith.muli %x, %u : i32
    printf.print_format "{}", %res : i32
    """
    pass_6_boxes, pass_6_mo = build_example(6, pass_6_mlir)

    mo.vstack([mo.hstack([pass_1_mo, pass_2_mo]), mo.md("<br>"), mo.hstack([pass_3_mo, pass_4_mo]), mo.md("<br>"), mo.hstack([pass_5_mo, pass_6_mo])])
    return (
        get_checkboxes_state,
        pass_1_boxes,
        pass_2_boxes,
        pass_3_boxes,
        pass_4_boxes,
        pass_5_boxes,
        pass_6_boxes,
    )


@app.cell
def _(
    get_checkboxes_state,
    mo,
    pass_1_boxes,
    pass_2_boxes,
    pass_3_boxes,
    pass_4_boxes,
    pass_5_boxes,
    pass_6_boxes,
):
    get_checkboxes_state

    boxess = (pass_1_boxes, pass_2_boxes, pass_3_boxes, pass_4_boxes, pass_5_boxes, pass_6_boxes)
    values = "_".join(
        "".join(str(int(box.value)) for box in boxes)
        for boxes in boxess
    )
    expected_values = "100_010_000_001_000_011"

    check_optimizations = "✅" if values == expected_values else "❌"
    values, check_optimizations

    mo.hstack((mo.md("✅ Correct!" if values == expected_values else "❌ At least one exercise is incorrect"),), justify="center")
    return (check_optimizations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Write non-optimal programs

    Can you write, for each pass, a program that would be optimized by it?
    """
    )
    return


@app.cell(hide_code=True)
def _(dce_example_tick, mo):
    mo.md(rf"""### Case 1: Program optimized by `dce` &nbsp; {dce_example_tick}""")
    return


@app.cell
def _(mo):
    reset_button10 = mo.ui.button(label="reset")
    reset_button10
    return (reset_button10,)


@app.cell
def _(mo, reset_button10):
    reset_button10

    _default = r"""%x = arith.constant 0 : i32
    printf.print_format "{}", %x : i32"""
    example_editor10 = mo.ui.code_editor(value=_default)

    example_editor10
    return (example_editor10,)


@app.cell
def _(
    Context,
    DeadCodeElimination,
    example_editor10,
    execute_and_catch_exceptions,
    mo,
    parse_mlir,
    xmo,
):
    def _execute():
        module = parse_mlir(example_editor10.value)
        str_module = str(module)
        DeadCodeElimination().apply(Context(), module)
        if str(module) == str_module:
            return False, mo.md("❌ The `dce` pass had no effects on the given program.")
        return True, mo.vstack([mo.md("✅ Some operations were removed!\n\nHere is the resulting program:\n"), xmo.module_md(module)])

    _correct, _output = execute_and_catch_exceptions(_execute)
    dce_example_tick = "✅" if _correct else "❌"
    _output
    return (dce_example_tick,)


@app.cell
def _(fold_example_tick, mo):
    mo.md(rf"""### Case 2: Program optimized by `constant-fold-interp` &nbsp; {fold_example_tick}""")
    return


@app.cell
def _(mo):
    reset_button12 = mo.ui.button(label="reset")
    reset_button12
    return (reset_button12,)


@app.cell
def _(mo, reset_button12):
    reset_button12

    _default = r"""%x = arith.constant 0 : i32
    printf.print_format "{}", %x : i32"""
    example_editor12 = mo.ui.code_editor(value=_default)

    example_editor12
    return (example_editor12,)


@app.cell
def _(
    ConstantFoldInterpPass,
    Context,
    example_editor12,
    execute_and_catch_exceptions,
    mo,
    parse_mlir,
    xmo,
):
    def _execute():
        module = parse_mlir(example_editor12.value)
        str_module = str(module)
        ConstantFoldInterpPass().apply(Context(), module)
        if str(module) == str_module:
            return False, mo.md("❌ The `constant-fold-interp` pass had no effects on the given program.")
        return True, mo.vstack([mo.md("✅ Some operations were removed!\n\nHere is the resulting program:\n"), xmo.module_md(module)])

    _correct, _output = execute_and_catch_exceptions(_execute)
    fold_example_tick = "✅" if _correct else "❌"
    _output
    return (fold_example_tick,)


@app.cell
def _(cse_example_tick, mo):
    mo.md(rf"""### Case 3: Program optimized by `cse` &nbsp; {cse_example_tick}""")
    return


@app.cell
def _(mo):
    reset_button11 = mo.ui.button(label="reset")
    reset_button11
    return (reset_button11,)


@app.cell
def _(mo, reset_button11):
    reset_button11

    _default = r"""%x = arith.constant 0 : i32
    printf.print_format "{}", %x : i32"""
    example_editor11 = mo.ui.code_editor(value=_default)

    example_editor11
    return (example_editor11,)


@app.cell
def _(
    CommonSubexpressionElimination,
    Context,
    example_editor11,
    execute_and_catch_exceptions,
    mo,
    parse_mlir,
    xmo,
):
    def _execute():
        module = parse_mlir(example_editor11.value)
        str_module = str(module)
        CommonSubexpressionElimination().apply(Context(), module)
        if str(module) == str_module:
            return False, mo.md("❌ The `cse` pass had no effects on the given program.")
        return True, mo.vstack([mo.md("✅ Some operations were removed!\n\nHere is the resulting program:\n"), xmo.module_md(module)])

    _correct, _output = execute_and_catch_exceptions(_execute)
    cse_example_tick = "✅" if _correct else "❌"
    _output
    return (cse_example_tick,)


@app.cell
def _(exercise4_tick, mo):
    mo.md(
        rf"""
    ### Case 4: All the above! {exercise4_tick}

    When we write a compilation pipeline, we use way more than a single pass. Can you find a program that gets optimized at each step of the pipeline? Here, the passes `dce`, `cse`, `constant-fold-interp, dce` are called in order. Note that `dce` is called twice in the pipeline.
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
    _default = """%x = arith.constant 0 : i32
    printf.print_format "{}", %x : i32"""

    example_editor4 = mo.ui.code_editor(language="rs", value=_default)
    pass_editor4 = mo.ui.code_editor(value="dce,cse,constant-fold-interp,dce", max_height=1, label="Passes:")

    example_editor4
    return example_editor4, pass_editor4


@app.cell(hide_code=True)
def _(
    example_editor4,
    execute_and_catch_exceptions,
    get_compilation_outputs_with_passes,
    mo,
    pass_editor4,
):
    def _execute():
        outputs4 = get_compilation_outputs_with_passes(example_editor4, pass_editor4, "mlir")
        return True, outputs4

    _, outputs4 = execute_and_catch_exceptions(_execute)

    _cell_result = mo.md("")
    correct4 = False
    if isinstance(outputs4, mo.Html):
        _cell_result = outputs4
    else:
        labels4, modules4 = zip(*outputs4)
        if (modules4[0].text != modules4[1].text) and (modules4[1].text != modules4[2].text) and (modules4[2].text != modules4[3].text) and (modules4[3] != modules4[4].text):
            correct4 = True
            _cell_result = mo.md("✅ All passes had an effect!")
        else:
            _cell_result = mo.md("❌ At least one pass had no effects! \n\nYou can look at how the program is being modified at each pass below:")

    exercise4_tick = "✅" if correct4 else "❌"
    _cell_result
    return exercise4_tick, labels4, outputs4


@app.cell
def _(mo):
    get_state4, set_state4 = mo.state(0)
    return get_state4, set_state4


@app.cell
def _(get_state4, labels4, mo, set_state4):
    slider4 = mo.ui.slider(start=0, stop=len(labels4) - 1, value=get_state4(), on_change=set_state4)
    return (slider4,)


@app.cell
def _(get_state4, labels4, mo, outputs4, set_state4):
    tabs4 = mo.ui.tabs(dict(outputs4), value=labels4[get_state4()], on_change=lambda k: set_state4(labels4.index(k)))
    return (tabs4,)


@app.cell
def _(mo, slider4, tabs4):
    mo.vstack((slider4, tabs4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adding new abstractions with dialects

    Let's now introduce lists in our language. For that, we add the following operations:

    * Creating a list from a range (`x..y`)
    * Getting the length of a list (`list.len()`)
    * Mapping a function over a list (`list.map(|x| x + 1)`)

    To represent them in MLIR, we create our own collection of operations and types, which is called a **dialect**.

    Try to write some programs using these features, and look at the MLIR output.
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

    _initial_code = r"""let a = 3;
    let b = a..10;
    b.len()"""

    example_editor7 = mo.ui.code_editor(language="rs", value=_initial_code, label="MLIR code:")

    example_editor7
    return (example_editor7,)


@app.cell(hide_code=True)
def _(example_editor7, to_mlir, xmo):
    outputs7 = xmo.module_md(to_mlir(example_editor7.value))
    outputs7
    return


@app.cell
def _(mo):
    get_state7, set_state7 = mo.state(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | What do you see?

    * Types in MLIR are displayed with the syntax `!dialect.type`.
    * User-defined operations, as well as the custom type all start with `list`, the dialect name.
    * `list.map` uses a region to represent the function to be applied on each element of the list. This region has an argument, terminated by `list.yield`.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(exercise20_tick, mo):
    mo.md(
        rf"""
    ## Lowering our abstractions to MLIR {exercise20_tick}

    Now, we need to compile our abstractions to abstractions defined by MLIR. From there, we can use existing MLIR passes to lower our code to LLVM.

    Here, we are compiling our dialect to the `tensor` and `scf` dialects, which can represent tensors and loops.

    As a last interactive example how a full compiler pipeline works, here is a code editor where you can write your own program, your own compiler pipeline, and where you can then explore the different stages of the compiler. As an exercise, can you manage to write a program that gets optimized at each level?

    The optimizations that you can use are the following:

    * `canonicalize` : It is the combination of constant folding, dead code elimination, and additionally dialect-specific optimizations
    * `cse` : Constant sub-expression elimination, the same pass as before
    * `lower-list-to-tensor` : Compiles the list abstraction to the `scf` and `tensor` dialects
    * `licm` (Loop Invariant Code Motion) : Hoist variables outside of loops when they do not depend on any variables inside the loop
    * `optimize-lists: Apply domain-specific optimizations on the `list` dialect
    """
    )
    return


@app.cell
def _(mo):
    reset_button20 = mo.ui.button(label="reset")
    reset_button20
    return (reset_button20,)


@app.cell(hide_code=True)
def _(mo, reset_button20):
    reset_button20

    _initial_code = r"""let a = 0..10;
    let c = a.map(|x| x + a.len());
    c"""

    example_editor20 = mo.ui.code_editor(language="rs", value=_initial_code)
    pass_editor20 = mo.ui.code_editor(value="optimize-lists,canonicalize,cse,lower-list-to-tensor,canonicalize,licm,cse", max_height=1)

    mo.vstack([example_editor20, pass_editor20])
    return example_editor20, pass_editor20


@app.cell
def _(
    Context,
    LowerListToTensor,
    OptimizeListOps,
    PassPipeline,
    example_editor20,
    execute_and_catch_exceptions,
    get_all_passes,
    lmo,
    pass_editor20,
    to_mlir,
):
    def _exec():
        _module = to_mlir(example_editor20.value)
        _all_passes = get_all_passes()
        _all_passes["lower-list-to-tensor"] = lambda: LowerListToTensor()
        _all_passes["optimize-lists"] = lambda: OptimizeListOps()
        PassPipeline.parse_spec(_all_passes, pass_editor20.value).apply(Context(), _module)
        return True, lmo.interp(_module)

    _, exec_res20 = execute_and_catch_exceptions(_exec)

    exec_res20
    return


@app.cell
def _(
    example_editor20,
    execute_and_catch_exceptions,
    get_compilation_outputs_with_passes,
    mo,
    pass_editor20,
):
    def _execute():
        outputs20 = get_compilation_outputs_with_passes(example_editor20, pass_editor20)
        return True, outputs20

    _, outputs20 = execute_and_catch_exceptions(_execute)

    _cell_result = mo.md("")
    correct20 = False
    if isinstance(outputs20, mo.Html):
        _cell_result = outputs20
        labels20 = []
        modules20 = []
    else:
        labels20, modules20 = zip(*outputs20)
        texts20 = {m.text for m in modules20}
        if len(texts20) == len(modules20):
            correct20 = True
            _cell_result = mo.md("✅ All passes had an effect!")
        else:
            _cell_result = mo.md("❌ At least one pass had no effects! \n\nYou can look at how the program is being modified at each pass below:")

    exercise20_tick = "✅" if correct20 else "❌"
    _cell_result
    return exercise20_tick, labels20, outputs20


@app.cell
def _(mo):
    get_state20, set_state20 = mo.state(0)
    return get_state20, set_state20


@app.cell
def _(get_state20, labels20, mo, set_state20):
    slider20 = mo.ui.slider(start=0, stop=len(labels20) - 1, value=get_state20(), on_change=set_state20)
    return (slider20,)


@app.cell
def _(get_state20, labels20, mo, outputs20, set_state20):
    tabs20 = mo.ui.tabs(dict(outputs20), value=labels20[get_state20()], on_change=lambda k: set_state20(labels20.index(k)))
    return (tabs20,)


@app.cell
def _(mo, slider20, tabs20):
    mo.vstack((slider20, tabs20))
    return


if __name__ == "__main__":
    app.run()
