import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from xdsl.dialects.arith import AddiOp
    from xdsl.dialects.builtin import IntegerType
    from xdsl.printer import Printer
    from xdsl.utils.test_value import TestSSAValue

    return AddiOp, IntegerType, Printer, TestSSAValue, mo


@app.cell
def _(mo):
    mo.md(
        r"""
        # Operations

        ## What is an operation

        An operation is the basic unit of execution. It usually reprents a computation that can be arbitrarily complex, or can represent information. An operation is an instantiation of an op, otherwise known as an operation definition.

        Example of ops are:

        * `arith.addi`, which adds two integer values together
        * `linalg.matmul`, which multiplies two matrices together
        * `func.func`, which defines a function

        Operations are composed of:

        * A name,
        * A list of operands, which are dynamic values taken as input
        * A list of results, which are dynamic values outputted by the operation
        * Properties, which are encoded as an attribute dictionary, which correspond to additional information operations may need (such as flags)
        * An attribute dictionary, which correspond to additional discardable information that users can attach to operations.
        * A list of regions, which correspond to code blocks that may be executed by the operation
        * A list of successors (basic block arguments), which are the possible block that the operation may give control flow to. Only terminator operations may have successors.

        ## Instantiating operations

        Each op is encoded as a Python class, and each operation is an instantiation of the class.
        """
    )
    return


@app.cell
def _(AddiOp, IntegerType, Printer, TestSSAValue):
    # Create two SSA values of type i32 for testing
    lhs = TestSSAValue(IntegerType(32))
    rhs = TestSSAValue(IntegerType(32))

    # Create an operation
    addi_op = AddiOp(lhs, rhs)

    # Print it
    print(addi_op)

    # Print it with the generic format
    Printer(print_generic_format=True).print(addi_op)
    return addi_op, lhs, rhs


@app.cell
def _(mo):
    mo.md(
        r"""Note that here, a default attribute was created for the `overflowFlags` property."""
    )
    return


if __name__ == "__main__":
    app.run()
