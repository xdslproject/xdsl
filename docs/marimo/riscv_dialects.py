import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.ir import Block, Region
    from xdsl.builder import Builder, InsertPoint
    from xdsl.dialects import builtin, riscv, riscv_cf, riscv_func
    from xdsl.printer import Printer
    from xdsl.parser import Parser
    from xdsl.context import Context
    import difflib
    return (
        Block,
        Builder,
        Context,
        InsertPoint,
        Parser,
        Printer,
        builtin,
        difflib,
        mo,
        riscv,
        riscv_cf,
        riscv_func,
        xmo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# RISC-V Dialects""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RISC-V

    The [RISC-V](https://riscv.org/) instruction set is a small, extensible set of instructions that is gaining in popularity in research and industry.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Click here for some resources on the RISC-V ISA and assembly format": (
                """
    * [Cheat Sheet](https://www.cl.cam.ac.uk/teaching/1617/ECAD+Arch/files/docs/RISCVGreenCardv8-20151013.pdf)
    * [Assembly Manual](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc)
    * [ISA Manual](https://github.com/riscv/riscv-isa-manual)
    * [Detailed Instruction Definition](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html)
    * [ABI](https://d3s.mff.cuni.cz/files/teaching/nswi200/202324/doc/riscv-abi.pdf)
    * [Formal Specification](https://github.com/riscv/sail-riscv)"""
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Simple Arithmetic in RISC-V""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here is a simple C function and its corresponding assembly:

    ```C
    int multiply_add(int a, int b, int c) {
        return a + b * c;
    }
    ```

    ```asm
    # Label corresponding to the function name
    # Arguments passed in a0, a1, a2 registers
    # Result expected to be stored in a0 register at the end of execution
    multiply_add:
        # a1 <- a2 * a1
        mul     a1, a2, a1
        # a0 <- a0 + a1
        add     a0, a0, a1
        # Assembly pseudo-operation that jumps to the caller-passed return address
        ret
    ```

    ([Compiler Explorer](https://godbolt.org/z/vscn4n1oh))
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A few aspects of assembly make it difficult to reason about directly.
    It doesn't track dependencies explicitly, making it difficult to know what the values in the inputs of an operation are, and whether its results will be used.
    Control flow is done via jumps to labels, such as the `multiply_add` above.
    In order to represent and reason about assembly-level code, we introduce a set of backend dialects: `riscv`, `riscv_func`, and `riscv_cf`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The `riscv` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `risv` dialect contains definitions for static information and operations.

    The four kinds of static information in RISC-V assembly are strings, integers, labels, and registers.
    The `riscv` dialect includes integer and float registers (`IntRegisterType` & `FloatRegisterType`).
    Both of these register classes encode both the information about the binary encoding of the register (e.g. `x0`, `x11`) and its pretty assembly name (e.g. `zero`, `a1`).

    The pretty assembly names reflect the RISC-V ABI. The registers `x0` to `x4` are specified to store some system information: `zero` for a register that always has the 0 value, `ra` to store the return address to jump to, `sp` for stack pointer, `gp` for global pointer, `tp` for thread pointer. The `a0-a7` registers are used to pass function arguments and results. The `s0-s11` registers are expected to have the same values before and after function calls. The `t0-t6` registers are temporary registers that are not expected to have the same values before and after function calls.
    """
    )
    return


@app.cell
def _(riscv):
    # Explictly constructing registers

    riscv.IntRegisterType.from_name("zero"), riscv.IntRegisterType.from_name("a1"), riscv.IntRegisterType.from_name("s2"), riscv.IntRegisterType.from_name("t3")
    return


@app.cell
def _(riscv):
    # Using the Registers helper

    riscv.Registers.ZERO, riscv.Registers.A1, riscv.Registers.S2, riscv.Registers.T3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We use these registers as the types of the operands and results of operations in the riscv dialect.
    Let's construct a dummy block with some operations in it:
    """
    )
    return


@app.cell
def _(Block, Builder, InsertPoint, Printer, riscv):
    block = Block(arg_types=(riscv.Registers.A0, riscv.Registers.A1))
    a0, a1 = block.args
    builder = Builder(InsertPoint.at_end(block))

    # Explicitly specify result registers
    addi_op = builder.insert(riscv.AddiOp(a0, 42, rd=riscv.Registers.A2))
    sub_op = builder.insert(riscv.SubOp(a0, a1, rd=riscv.Registers.A3))
    mul_op = builder.insert(riscv.MulOp(addi_op.rd, sub_op.rd, rd=riscv.Registers.A4))

    Printer().print_block(block)
    return (block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that the _i_ in the `addi` operation above stands for _immediate_, or a constant value in the assembly.
    These can be integers or labels, in which case the assembler will resolve the immediate to the value referred to by the label.

    Almost all the operations in the riscv dialect correspond to the instructions in the RISC-V ISA.
    RISC-V assembly has operations that correspond 1:1 with instructions, as well as pseudo-operations that are syntactic sugar for other operations, such as the `nop` instruction that desugars to `addi x0, x0, 0`.

    [The RISC-V Assembly Manual](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc#a-listing-of-standard-risc-v-pseudoinstructions) has the list of pseudo-instructions in RISC-V.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Printing Assembly""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""All operations in the `riscv` dialect implement an `assembly_line` function that communicates how they should be displayed in assembly:""")
    return


@app.cell
def _(block):
    for op in block.ops:
        print(op.assembly_line())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Functions and the `riscv_func` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When compiling a function for certain hardware, an Application Binary Interface (ABI) specifies things like which registers will hold the function arguments when the function begins, which registers will hold the results when the function ends, which registers are overridden, etc.

    We use the `riscv_func` dialect to represent RISC-V-specific functions that encode these calling convention considerations in the IR.

    Here is our `mul` function from above:
    """
    )
    return


@app.cell
def _():
    mul_ir = """
    riscv_func.func @mul(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      %res = riscv.mul %num, %num : (!riscv.reg<a0>, !riscv.reg<a0>) -> !riscv.reg<a0>
      riscv_func.return %res : !riscv.reg<a0>
    }
    """
    return (mul_ir,)


@app.cell
def _(Parser, ctx, mul_ir, riscv, xmo):
    mul_module = Parser(ctx, mul_ir).parse_module()

    xmo.asm_html(riscv.riscv_code(mul_module))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Unstructured Control Flow and the `riscv_cf` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The C programming language allows for both _structured_ control flow with the `if`, `for`, and `while` constructs, and _unstructured_ flow using goto. The two functions below lower to exactly the same assembly:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```C
    int switch1(int a, int b, int c) {
        if (a) {
            return b;
        } else {
            return c;
        }
    }

    int switch2(int a, int b, int c) {
        if (!a) {
            goto c_label;
        }
        return b;
    c_label:
        return c;
    }
    ```

    ([Compiler Explorer](https://godbolt.org/z/Mzh4ojG3r))
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    At the assembly level, the code structure looks much more like the code in `switch2`.

    To represent jumps in the assembly, we use blocks and successors.
    For example, the [beq](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq) instruction, which jumps by (or _breaks to_) the specified offset (or to the specified label) if the values in the source registers are equal, is represented with the `riscv.BeqOp` operation:
    """
    )
    return


@app.cell(hide_code=True)
def _(Parser, ctx, xmo):
    switch_ir = """\
    riscv_func.func @switch(%a : !riscv.reg<a0>, %b : !riscv.reg<a1>, %c : !riscv.reg<a2>) -> !riscv.reg<a0> {
      %zero = riscv.get_register : !riscv.reg<zero>
      riscv_cf.beq %a : !riscv.reg<a0>, %zero : !riscv.reg<zero>, ^bb2(), ^bb1()
    ^bb1():
      %res_b = riscv.mv %b : (!riscv.reg<a1>) -> !riscv.reg<a0>
      riscv_func.return %res_b : !riscv.reg<a0>
    ^bb2():
      riscv.label "c_label"
      %res_c = riscv.mv %c : (!riscv.reg<a2>) -> !riscv.reg<a0>
      riscv_func.return %res_c : !riscv.reg<a0>
    }
    """
    switch_module = Parser(ctx, switch_ir).parse_module()
    switch_module.verify()
    xmo.module_html(switch_module)
    return (switch_module,)


@app.cell
def _(riscv, switch_module):
    # We can print this IR as assembly
    switch_asm = riscv.riscv_code(switch_module)
    print(switch_asm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that there are a few discrepancies between the assembly and the IR.
    While in the IR we specified both the "then" and the "else" successors, in assembly the block implicitly falls through.
    We represent this by adding a verification method to the jump operations that checks that the "else" block is the one immediately following the conditional jump.
    The second discrepancy is that the conditional jump in assembly specifies the label to jump to.
    During assembly printing, the `BeqOp` looks at the first operation in the block and, if it finds a `LabelOperation`, prints the corresponding value.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exercise: Fib IR""")
    return


@app.cell(hide_code=True)
def _():
    fib_text = """\
    # The label corresponding to the fib function
    fib:
        bge zero, a0, .LBB1_4
        li a2, 1
        li a3, 1
    # Label inserted by Clang corresponding to the loop body
    .LBB1_2:
        add a4, a2, a3
        addi a0, a0, -1
        mv a1, a3
        mv a2, a3
        mv a3, a4
        # Jump to beginning of loop body if need to do more work
        bne zero, a0, .LBB1_2
        # Moves the result value to the first integer return register, as specified by ABI
        mv a0, a1
        ret
    # Label inserted by Clang corresponding to the case where the input is <= 0, can just return 1
    .LBB1_4:
        li a0, 1
        ret
    """
    return (fib_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this exercise, we will rewrite the following function to assembly-level IR:

    ```C
    // Assume n non-negative
    int fib(int n) {
        int a = 1;
        int b = 1;
        for (int i = 0; i < n; ++i) {
            int c = a + b;
            a = b;
            b = c;
        }
        return a;
    }
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(fib_text, mo):
    mo.md(
        fr"""
    This is the expected assembly:

    ```asm
    {fib_text}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(comment_only_line, fib_text):
    fib_cleaned = "\n".join(line for line in fib_text.splitlines() if not comment_only_line.findall(line))
    return (fib_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    fib_editor = mo.ui.code_editor("""\
    riscv_func.func @fib(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      %zero = riscv.get_register : !riscv.reg<zero>
      riscv_cf.bge %zero: !riscv.reg<zero>, %num :!riscv.reg<a0>, ^bb4(), ^bb1()
    ^bb1():
      %a_init = riscv.li 1 : !riscv.reg<a2>
      %b_init = riscv.li 1 : !riscv.reg<a3>
      riscv_cf.branch ^bb2 (%num : !riscv.reg<a0>, %a_init : !riscv.reg<a2>, %b_init : !riscv.reg<a3>)
    ^bb2(%i : !riscv.reg<a0>, %a_in : !riscv.reg<a2>, %b_in : !riscv.reg<a3>):
      riscv.label ".LBB1_2"
      %sum = riscv.li 2 : !riscv.reg<a4>
      %i_next = riscv.li 3 : !riscv.reg<a0>
      %temp = riscv.li 4 : !riscv.reg<a1>
      %a_next = riscv.li 5 : !riscv.reg<a2>
      %b_next = riscv.li 6 : !riscv.reg<a3>
      riscv_cf.bne %zero: !riscv.reg<zero>, %i_next : !riscv.reg<a0>, ^bb2(%i_next : !riscv.reg<a0>, %a_next : !riscv.reg<a2>, %b_next : !riscv.reg<a3>), ^bb3()
    ^bb3():
      %res = riscv.mv %temp : (!riscv.reg<a1>) -> !riscv.reg<a0>
      riscv_func.return %num : !riscv.reg<a0>
    ^bb4():
      riscv.label ".LBB1_4"
      %res_early = riscv.li 1 : !riscv.reg<a0>
      riscv_func.return %res_early : !riscv.reg<a0>
    }""", language="javascript")
    return (fib_editor,)


@app.cell(hide_code=True)
def _(fib_editor, mo):
    mo.md(
        f"""
    Modify the following IR to reduce the diff below:

    {fib_editor}
    """
    )
    return


@app.cell(hide_code=True)
def _(Parser, ctx, difflib, fib_cleaned, fib_editor, mo, riscv):
    try:
        _module = Parser(ctx, fib_editor.value).parse_module()
        _module.verify()
        _asm = riscv.riscv_code(_module)
        _diff = "\n".join(difflib.ndiff(_asm.splitlines(), fib_cleaned.splitlines()))
        _res = f"""
    ```asm
    {_diff}
    ```
    """
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    mo.md(f"""
    Diff:

    {_res}
    """)
    return


@app.cell(hide_code=True)
def _():
    # Solution

    _ = """\
    riscv_func.func @fib(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      %zero = riscv.get_register : !riscv.reg<zero>
      riscv_cf.bge %zero: !riscv.reg<zero>, %num :!riscv.reg<a0>, ^bb4(), ^bb1()
    ^bb1():
      %a_init = riscv.li 1 : !riscv.reg<a2>
      %b_init = riscv.li 1 : !riscv.reg<a3>
      riscv_cf.branch ^bb2 (%num : !riscv.reg<a0>, %a_init : !riscv.reg<a2>, %b_init : !riscv.reg<a3>)
    ^bb2(%i : !riscv.reg<a0>, %a_in : !riscv.reg<a2>, %b_in : !riscv.reg<a3>):
      riscv.label ".LBB1_2"
      %sum = riscv.add %a_in, %b_in : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a4>
      %i_next = riscv.addi %i, -1 : (!riscv.reg<a0>) -> !riscv.reg<a0>
      %temp = riscv.mv %b_in : (!riscv.reg<a3>) -> !riscv.reg<a1>
      %a_next = riscv.mv %b_in : (!riscv.reg<a3>) -> !riscv.reg<a2>
      %b_next = riscv.mv %sum : (!riscv.reg<a4>) -> !riscv.reg<a3>
      riscv_cf.bne %zero: !riscv.reg<zero>, %i_next : !riscv.reg<a0>, ^bb2(%i_next : !riscv.reg<a0>, %a_next : !riscv.reg<a2>, %b_next : !riscv.reg<a3>), ^bb3()
    ^bb3():
      %res = riscv.mv %temp : (!riscv.reg<a1>) -> !riscv.reg<a0>
      riscv_func.return %num : !riscv.reg<a0>
    ^bb4():
      riscv.label ".LBB1_4"
      %res_early = riscv.li 1 : !riscv.reg<a0>
      riscv_func.return %res_early : !riscv.reg<a0>
    }
    """
    return


@app.cell(hide_code=True)
def _(Context, builtin, riscv, riscv_cf, riscv_func):
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(riscv_cf.RISCV_Cf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    return (ctx,)


@app.cell(hide_code=True)
def _():
    import re

    comment_only_line = re.compile(r"^\s*#.*$")
    return (comment_only_line,)


if __name__ == "__main__":
    app.run()
