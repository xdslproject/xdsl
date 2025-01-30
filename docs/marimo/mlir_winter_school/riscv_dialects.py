import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.ir import Block
    from xdsl.builder import Builder, InsertPoint
    from xdsl.dialects import riscv
    from xdsl.printer import Printer
    return Block, Builder, InsertPoint, Printer, mo, riscv, xmo


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
        * [Assembly Manual](https://github.com/riscv-non-isa/riscv-asm-manual)
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
        """
    )
    return


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

        # The label corresponding to the fib function
        fib:
                blez    a0, .LBB1_4
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
                bnez    a0, .LBB1_2
                # Moves the result value to the first integer return register, as specified by ABI
                mv      a0, a1
                ret
        # Label inserted by Clang corresponding to the case where the input is <= 0, can just return 1
        .LBB1_4:
                li      a0, 1
                ret
        ```

        ([Compiler Explorer](https://godbolt.org/z/bedsKMWMc))
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
    addi_op = builder.insert(riscv.AddiOp(a0, a1, rd=riscv.Registers.A2))
    sub_op = builder.insert(riscv.SubOp(a0, a1, rd=riscv.Registers.A3))

    # Don't specify result registers
    mul_op = builder.insert(riscv.MulOp(addi_op.rd, sub_op.rd, rd=riscv.Registers.UNALLOCATED_INT))

    Printer().print_block(block)
    return a0, a1, addi_op, block, builder, mul_op, sub_op


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Unstructured Control Flow and the `riscv_cf` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Structured Control Flow and the `riscv_scf` Dialect""")
    return


if __name__ == "__main__":
    app.run()
