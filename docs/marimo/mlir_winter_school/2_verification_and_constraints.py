import marimo

__generated_with = "0.10.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Verification and Constraints

        ## Verification

        Attributes provide an API to check at compile time that they have been constructed correctly.
        This check is run during initialization.
        """
    )
    return


@app.cell
def _():
    from xdsl.dialects.builtin import IntegerType

    IntegerType(32), IntegerType(64)
    return (IntegerType,)


@app.cell
def _(mo):
    mo.md(r"""Each attribute definition defines a set of accepted parameters. `IntegerType` for instance has only a single bitwidth parameter that should be a positive integer. When a set of parameters is invalid, the attribute should return a `VerifyException` upon instantiation.""")
    return


@app.cell
def _(IntegerType):
    from xdsl.utils.exceptions import VerifyException

    try:
        IntegerType(-1)
    except VerifyException as e:
        print(e)
    return (VerifyException,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Given an `Attribute` variable, the python `isinstance` function allows to match a specific attribute:""")
    return


@app.cell
def _(Attribute, IntegerType):
    from xdsl.dialects.builtin import IndexType

    def print_if_integer_type(attr: Attribute):
        if isinstance(attr, IntegerType):
            print(attr)
        else:
            print("Not an integer type")

    print_if_integer_type(IntegerType(32))
    print_if_integer_type(IndexType())
    return IndexType, print_if_integer_type


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
