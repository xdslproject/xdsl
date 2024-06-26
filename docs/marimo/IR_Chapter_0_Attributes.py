import marimo

__generated_with = "0.6.10"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    from xdsl.dialects.builtin import IntAttr

    mo.md(f"""
    # IR Chapter 1: Attributes

    In xDSL, Attributes carry information known at compile time.
    For example, `{IntAttr.__name__}` is an attribute representing an integer.
    """)
    return IntAttr,


if __name__ == "__main__":
    app.run()
