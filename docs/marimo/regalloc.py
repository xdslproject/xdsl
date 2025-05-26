import marimo

__generated_with = "0.13.6"
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
    from xdsl.context import Context
    import difflib
    from xdsl.backend.register_type import RegisterType
    return RegisterType, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Register Allocation in RISC-V""")
    return


@app.cell(hide_code=True)
def _(RegisterType, mo):
    mo.md(
        rf"""
    ## Register Allocation

    [Register allocation](https://en.wikipedia.org/wiki/Register_allocation) is the process of assigning values to registers.
    This notebook describes the implementation of register allocation in xDSL, with our RISC-V dialect as an example.
    Our implementation is a work in progress, and is currently limited in ways that we'll describe below.

    The pass works on values with a type that inherits from [`{RegisterType.__name__}`](https://xdsl.readthedocs.io/stable/reference/backend/register_type/#xdsl.backend.register_type.RegisterType).
    Values of this type may be _allocated_, meaning that the value must be stored in a specific register at run time, or _unallocated_ otherwise.
    During the execution of the program, a value with an allocated type being initialised corresponds to a write to a register, and a value being used corresponds to a register read.
    This means that, if a value with a given type is initialised between the initialisation and use of another value of the same type, 

    The objective of the pass is to rewrite the IR from containing unallocated values to only containing allocated ones, with the condition that.

    In practice, this isn't always possible, as the number of 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Register Allocation in `riscv`


    """
    )
    return


if __name__ == "__main__":
    app.run()
