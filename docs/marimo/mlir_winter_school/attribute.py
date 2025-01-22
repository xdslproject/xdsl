import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from xdsl.dialects.builtin import IndexType, IntegerType
    from xdsl.ir import Attribute
    from xdsl.utils.exceptions import VerifyException
    return Attribute, IndexType, IntegerType, VerifyException, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Attributes (and types)

        ## What is an attribute

        Attributes store static information that can be used in MLIR IR.
        Attributes are present in both the operation properties and attribute dictionary, and encode the type for each SSA Value.

        Examples of attributes are:

        * `i32`, a signless integer type of bitwidth 32
        * `!llvm.void`, representing `void` in the LLVM dialect
        * `4 : i32`, the constant 4 encoded on 32 bits
        * `memref<5xi32>`, a memory reference to 5 32-bits integers

        In particular, types are a subset of attributes, and only types are allowed to encode SSA value types.

        ## Instantiating attributes

        Each attribute is an instance of a given attribute definition. Each attribute definition is represented as a single Python type, which decides how the type parameters are stored in memory. For instance, `i32` is an instance of an `IntegerType` with `32` for its bitwidth parameter.
        """
    )
    return


@app.cell
def _(IntegerType):
    print(IntegerType(32))
    print(IntegerType(64))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Each attribute definition defines a set of accepted parameters. `IntegerType` for instance has only a single bitwidth parameter that should be a positive integer. When a set of parameters is invalid, the attribute should return a `VerifyException` upon instantiation.""")
    return


@app.cell
def _(IntegerType, VerifyException):
    try:
        print(IntegerType(-1))
    except VerifyException as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Given an `Attribute` variable, the python `isinstance` function allows to match a specific attribute:""")
    return


@app.cell
def _(Attribute, IndexType, IntegerType):
    def print_if_integer_type(attr: Attribute):
        if isinstance(attr, IntegerType):
            print(attr)
        else:
            print("Not an integer type")

    print_if_integer_type(IntegerType(32))
    print_if_integer_type(IndexType())
    return (print_if_integer_type,)


if __name__ == "__main__":
    app.run()
