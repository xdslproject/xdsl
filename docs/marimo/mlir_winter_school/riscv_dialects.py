import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.ir import Block, Region
    from xdsl.builder import Builder, InsertPoint
    from xdsl.dialects import builtin, riscv, riscv_cf, riscv_func
    from xdsl.printer import Printer
    from xdsl.parser import Parser
    from xdsl.context import MLContext
    return (
        Block,
        Builder,
        InsertPoint,
        MLContext,
        Parser,
        Printer,
        Region,
        builtin,
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

        Here are some resources on the RISC-V ISA and assembly format:

        * [Cheat Sheet](https://www.cl.cam.ac.uk/teaching/1617/ECAD+Arch/files/docs/RISCVGreenCardv8-20151013.pdf)
        * [Assembly Manual](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc)
        * [ISA Manual](https://github.com/riscv/riscv-isa-manual)
        * [Detailed Instruction Definition](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html)
        * [ABI](https://d3s.mff.cuni.cz/files/teaching/nswi200/202324/doc/riscv-abi.pdf)
        * [Formal Specification](https://github.com/riscv/sail-riscv)
        """
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
        int fused_multiply_add(int a, int b, int c) {
            return a * b + c;
        }
        ```

        ```asm
        # Label corresponding to the function name
        # Arguments passed in a0, a1, a2 registers
        # Result expected to be stored in a0 register at the end of execution
        fused_multiply_add:
            # a0 <- a0 * a1
            mul     a0, a1, a0
            # a0 <- a0 + a2
            add     a0, a0, a2
            # Assembly pseudo-operation that jumps to the caller-passed return address
            ret
        ```

        ([Compiler Explorer](https://godbolt.org/z/veWo8rbnK))
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A few aspects of assembly make it difficult to reason about directly.
        It doesn't track dependencies explicitly, making it difficult to know what the values in the inputs of an operation are, and whether its results will be used.
        Control flow is done via jumps to labels, such as the `fused_multiply_add` above.
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
        In order to reason about and represent these assembly-level operations, we define the `riscv` dialect.

        The four kinds of static information in RISC-V assembly are strings, integers, labels, and registers.
        The `riscv` dialect includes integer and float registers (`IntRegisterType` & `FloatRegisterType`).
        Both of these register classes encode both the information about the binary encoding of the register (e.g. `x0`, `x11`) and its pretty assembly name (e.g. `zero`, `a1`).
        """
    )
    return


@app.cell
def _(riscv):
    # Explictly constructing registers

    riscv.IntRegisterType("zero"), riscv.IntRegisterType("a1")
    return


@app.cell
def _(riscv):
    # Using the Registers helper

    riscv.Registers.ZERO, riscv.Registers.A1
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

    # Don't specify result registers
    mul_op = builder.insert(riscv.MulOp(addi_op.rd, sub_op.rd, rd=riscv.Registers.A4))

    Printer().print_block(block)
    return a0, a1, addi_op, block, builder, mul_op, sub_op


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    return (op,)


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
    return (mul_module,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Unstructured Control Flow and the `riscv_cf` Dialect""")
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
        To represent jumps in the assembly, we use blocks and successors.
        For example, the [beq](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq) instruction, which jumps by the specified offset (or to the specified label) if the values in the source registers are equal, is represented with the `riscv.BeqOp` operation:
        """
    )
    return


@app.cell
def _(Block, Builder, InsertPoint, Printer, Region, riscv, riscv_cf):
    _block0 = Block(arg_types=(riscv.Registers.A0, riscv.Registers.A1, riscv.Registers.A2))
    _block1 = Block(arg_types=(riscv.Registers.A2,))
    _block2 = Block(arg_types=(riscv.Registers.A2,))
    beq_region = Region((_block0, _block1, _block2))
    _a0, _a1, _a2 = _block0.args

    _builder = Builder(InsertPoint.at_end(_block0))
    _builder.insert(riscv_cf.BeqOp(_a0, _a1, [_a2], [_a2], _block2, _block1))

    _builder.insertion_point = InsertPoint.at_end(_block1)
    _builder.insert(riscv.LabelOp("else"))
    _builder.insert(riscv.AddiOp(_block1.args[0], 42, rd=riscv.Registers.A3))

    _builder.insertion_point = InsertPoint.at_end(_block2)
    _builder.insert(riscv.LabelOp("then"))
    _builder.insert(riscv.AddiOp(_block2.args[0], -42, rd=riscv.Registers.A3))

    Printer().print_region(beq_region)
    return (beq_region,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can print this IR as assembly:""")
    return


@app.cell
def _(beq_region):
    for _block in beq_region.blocks:
        for _op in _block.ops:
            print(_op.assembly_line())
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
    mo.md(fr"""

    ```asm
    {fib_text}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(comment_only_line, fib_text):
    fib_cleaned = "\n".join(line for line in fib_text.splitlines() if not comment_only_line.findall(line))
    return (fib_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    fib_editor = mo.ui.code_editor("""\
    riscv_func.func @fib(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      riscv_func.return %num : !riscv.reg<a0>
    }""", language="javascript")
    return (fib_editor,)


@app.cell(hide_code=True)
def _(fib_editor, mo):
    mo.md(f"""
    Modify the following IR to reduce the diff below:

    {fib_editor}

    """)
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


@app.cell
def _():
    import difflib
    return (difflib,)


@app.cell
def _():
    # Solution

    _ = """\
    riscv_func.func @fib(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      %zero = riscv.get_register : !riscv.reg<zero>
      riscv_cf.bge %zero: !riscv.reg<zero>, %num :!riscv.reg<a0>, ^4(), ^1()
    ^1():
      %a_init = riscv.li 1 : !riscv.reg<a2>
      %b_init = riscv.li 1 : !riscv.reg<a3>
      riscv_cf.branch ^2 (%num : !riscv.reg<a0>, %a_init : !riscv.reg<a2>, %b_init : !riscv.reg<a3>)
    ^2(%i : !riscv.reg<a0>, %a_in : !riscv.reg<a2>, %b_in : !riscv.reg<a3>):
      riscv.label ".LBB1_2"
      %sum = riscv.add %a_in, %b_in : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a4>
      %i_next = riscv.addi %i, -1 : (!riscv.reg<a0>) -> !riscv.reg<a0>
      %temp = riscv.mv %b_in : (!riscv.reg<a3>) -> !riscv.reg<a1>
      %a_next = riscv.mv %b_in : (!riscv.reg<a3>) -> !riscv.reg<a2>
      %b_next = riscv.mv %sum : (!riscv.reg<a4>) -> !riscv.reg<a3>
      riscv_cf.bne %zero: !riscv.reg<zero>, %num :!riscv.reg<a0>, ^2(), ^3()
    ^3():
      %res = riscv.mv %temp : (!riscv.reg<a1>) -> !riscv.reg<a0>
      riscv_func.return %num : !riscv.reg<a0>
    ^4():
      riscv.label ".LBB1_4"
      %res_early = riscv.li 1 : !riscv.reg<a0>
      riscv_func.return %res_early : !riscv.reg<a0>
    }
    """
    return


@app.cell(hide_code=True)
def _(MLContext, builtin, riscv, riscv_cf, riscv_func):
    ctx = MLContext()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(riscv_cf.RISCV_Cf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    return (ctx,)


@app.cell(hide_code=True)
def _():
    import re

    comment_only_line = re.compile(r"^\s*#.*$")
    return comment_only_line, re


if __name__ == "__main__":
    app.run()
