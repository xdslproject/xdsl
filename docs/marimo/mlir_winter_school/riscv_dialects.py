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
        * [Assembly Manual](https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_ASM.pdf))
        * [ISA Manual](https://github.com/riscv/riscv-isa-manual)
        * [Detailed Instruction Definition](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html)
        * [ABI](https://d3s.mff.cuni.cz/files/teaching/nswi200/202324/doc/riscv-abi.pdf)
        * [Formal Specification](https://github.com/riscv/sail-riscv)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Some Example Functions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here are some functions in C:

        ```C
        int square(int num) {
            return num * num;
        }
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

        ([Compiler Explorer](https://godbolt.org/z/bedsKMWMc))
        """
    )
    return


@app.cell(hide_code=True)
def _():
    fib_text = """
    # The label corresponding to the fib function
    fib:
            ble     zero, a0, .LBB1_4
            li      a2, 1
            li      a3, 1
    # Label inserted by Clang corresponding to the loop body
    .LBB1_2:
            add     a4, a2, a3
            addi    a0, a0, -1
            mv      a1, a3
            mv      a2, a3
            mv      a3, a4
            # Jump to beginning of loop body if need to do more work
            bne     zero, a0, .LBB1_2
            # Moves the result value to the first integer return register, as specified by ABI
            mv      a0, a1
            ret
    # Label inserted by Clang corresponding to the case where the input is <= 0, can just return 1
    .LBB1_4:
            li      a0, 1
            ret
    """
    return (fib_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And their corresponding RISC-V assembly:

        ```asm
        # The label corresponding to the square function
        square:
                # Results and arguments are stored in a0-a7
                mul     a0, a0, a0
                # Assembly pseudo-operation that jumps to the caller-passed return address
                ret
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
    addi_op = builder.insert(riscv.AddiOp(a0, a1, rd=riscv.Registers.A2))
    sub_op = builder.insert(riscv.SubOp(a0, a1, rd=riscv.Registers.A3))

    # Don't specify result registers
    mul_op = builder.insert(riscv.MulOp(addi_op.rd, sub_op.rd, rd=riscv.Registers.A4))

    Printer().print_block(block)
    return a0, a1, addi_op, block, builder, mul_op, sub_op


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Printing Assembly""")
    return


@app.cell
def _(block):
    for op in block.ops:
        print(op.assembly_line())
    return (op,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Functions and the `riscv_cf` Dialect""")
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
    mo.md(r"""As an exercise, write the MLIR IR corresponding to the fib function above:""")
    return


@app.cell(hide_code=True)
def _(comment_only_line, fib_text, mo):
    fib_cleaned = "\n".join(line for line in fib_text.splitlines() if not comment_only_line.findall(line))
    mo.md(f"""
    Goal:

    ```asm
    {fib_cleaned}
    ```
    """)
    return (fib_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    fib_editor = mo.ui.code_editor("""\
    riscv_func.func @fib(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      riscv_func.return %num : !riscv.reg<a0>
    }""", language="javascript")
    return (fib_editor,)


@app.cell
def _(Parser, ctx, fib_editor, mo, riscv):
    try:
        _module = Parser(ctx, fib_editor.value).parse_module()
        _module.verify()
        _res = f"""
    Result:

    ```asm
    {riscv.riscv_code(_module)}
    ```
    """
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"
        # raise e from e

    mo.md(f"""
    IR:

    {fib_editor}

    {_res}

    """)
    return


@app.cell
def _():
    # Solution

    _ = """
    riscv_func.func @fib(%num : !riscv.reg<a0>) -> !riscv.reg<a0> {
      %zero = riscv.get_register : !riscv.reg<zero>
      riscv_cf.bge %zero: !riscv.reg<zero>, %num :!riscv.reg<a0>, ^4(), ^1()
    ^1():
      %a_init = riscv.li 1 : !riscv.reg<a2>
      %b_init = riscv.li 1 : !riscv.reg<a3>
      riscv_cf.branch ^2 (%num : !riscv.reg<a0>, %a_init : !riscv.reg<a2>, %b_init : !riscv.reg<a3>)
    ^2(%i : !riscv.reg<a0>, %a_in : !riscv.reg<a2>, %b_in : !riscv.reg<a3>):
      riscv.label ".LBB1_2"
      %c = riscv.add %a_in, %b_in : (!riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a4>
      %i_next = riscv.addi %i, -1 : (!riscv.reg<a0>) -> !riscv.reg<a0>
      // TODO
      riscv_cf.bne %zero: !riscv.reg<zero>, %num :!riscv.reg<a0>, ^2(), ^3()
    ^3():
      riscv_func.return %num : !riscv.reg<a0>
    ^4():
      riscv.label ".LBB1_4"
      riscv_func.return %num : !riscv.reg<a0>
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
