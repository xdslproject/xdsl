import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Assembly-Level Structured Control Flow""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Structured Control Flow and the `riscv_scf` Dialect""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In 1968, Edsger Dijkstra wrote a famous letter called [_Go To Statement Considered Harmful_](https://dl.acm.org/doi/10.1145/362929.362947).
        In it he argues that it is better to avoid arbitrary jumps as they make it harder for programmers to reason about the code they are dealing with.
        It turns out that structure in code also makes it easier to write compilers that reason about it.
        We can use the `riscv_scf` dialect to represent structured control flow at the assembly level.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using the `riscv_scf` operations lets us represent this `fib` function:

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
