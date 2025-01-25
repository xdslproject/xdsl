import marimo

__generated_with = "0.10.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Generating and Manipulating Intermediate Representations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A Simple Arithmetic DSL

        In this notebook, we'll explore how to create and manipulate simple arithmetic expressions, defined in the following minimal DSL:
        """
    )
    return


@app.cell
def _():
    from dataclasses import dataclass
    from abc import ABC, abstractmethod

    @dataclass(frozen=True)
    class Expr:

        @abstractmethod
        def eval(self, ctx: dict[str, int]) -> int:
            ...

        def __add__(self, other) -> "Expr":
            return Add(self, other)

        def __sub__(self, other) -> "Expr":
            return Sub(self, other)

    @dataclass(frozen=True)
    class Constant(Expr):
        val: int

        def eval(self, ctx: dict[str, int]) -> int:
            return self.val

        def __str__(self):
            return str(self.val)

    @dataclass(frozen=True)
    class Var(Expr):
        name: str

        def eval(self, ctx: dict[str, int]) -> int:
            return ctx[self.name]

        def __str__(self):
            return self.name

    @dataclass(frozen=True)
    class Add(Expr):
        lhs: Expr
        rhs: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            return self.lhs.eval(ctx) + self.rhs.eval(ctx)

        def __str__(self):
            return f"{self.lhs} + {self.rhs}"

    @dataclass(frozen=True)
    class Mul(Expr):
        lhs: Expr
        rhs: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            return self.lhs.eval(ctx) * self.rhs.eval(ctx)

        def __str__(self):
            return f"{self.lhs} * {self.rhs}"

    @dataclass(frozen=True)
    class Sub(Expr):
        lhs: Expr
        rhs: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            return self.lhs.eval(ctx) - self.rhs.eval(ctx)

        def __str__(self):
            return f"{self.lhs} - {self.rhs}"

    @dataclass(frozen=True)
    class Mod(Expr):
        lhs: Expr
        rhs: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            return self.lhs.eval(ctx) % self.rhs.eval(ctx)

        def __str__(self):
            return f"{self.lhs} % {self.rhs}"

    @dataclass(frozen=True)
    class Div(Expr):
        lhs: Expr
        rhs: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            return self.lhs.eval(ctx) / self.rhs.eval(ctx)

        def __str__(self):
            return f"{self.lhs} / {self.rhs}"

    @dataclass(frozen=True)
    class If(Expr):
        cond: Expr
        t: Expr
        e: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            return self.t.eval(ctx) if self.cond.eval(ctx) else self.e.eval(ctx)

        def __str__(self):
            return f"{self.t} if {self.cond} else {self.e}"

    # @dataclass(frozen=True)
    # class Lambda(Expr):
    #     args: tuple[str, ...]
    #     body: Expr

    #     def eval(self, ctx: dict[str, int]) -> int:
    #         return self.body.eval({arg_name: arg_val for arg_name, arg_val in zip(self.args, args, strict=True)} | ctx)

    #     def __str__(self):
    #         args = " ".join(self.args)
    #         return "{" + f"({args}) in {self.body}" + "}"

    @dataclass(frozen=True)
    class Fold(Expr):
        n: Expr
        init: Expr
        args: tuple[str, str]
        body: Expr

        def eval(self, ctx: dict[str, int]) -> int:
            acc = self.init.eval(ctx)
            for i in range(self.n.eval(ctx)):
                acc = self.body.eval({f"{self.args[0]}": i, f"{self.args[1]}": acc})
            return acc

        def __str__(self):
            return f'fold n={self.n} init={self.init} ' "{" f"({self.args[0]}, {self.args[1]}) in {self.body}" "}"

    @dataclass
    class Func:
        name: str
        args: tuple[str, ...]
        body: Expr

        def eval(self, *args: int) -> int:
            arg_ctx = {arg_name: arg_val for arg_name, arg_val in zip(self.args, args, strict=True)}
            return self.body.eval(arg_ctx)

        def __str__(self):
            args = " ".join(self.args)
            return f"let {self.name} {args} = {self.body}"
    return (
        ABC,
        Add,
        Constant,
        Div,
        Expr,
        Fold,
        Func,
        If,
        Mod,
        Mul,
        Sub,
        Var,
        abstractmethod,
        dataclass,
    )


@app.cell
def _(Add, Constant, Fold, Func, If, Mul, Var, mo):
    fact = Func(
        "fact",
        ("n",),
        If(
            Var("n"),
            Fold(
                Var("n"),
                Constant(1),
                ("i", "acc"),
                Mul(Var("acc"), Add(Var("i"), Constant(1)))
            ),
            Constant(1)
        )
    )

    mo.md(f"""
    This simple DSL lets us express simple functions, such as the factorial function:

    `{fact}`

    We can evaluate this function on a number of inputs:

    ```
    fact.eval(1) = {fact.eval(1)}
    fact.eval(2) = {fact.eval(2)}
    fact.eval(3) = {fact.eval(3)}
    fact.eval(4) = {fact.eval(4)}
    ```
    """)
    return (fact,)


@app.cell
def _(mo, triangle):
    mo.md(fr"""
    **Exercise 1: Triangle Numbers**

    Modify the definition of `triangle` below to compute the sum from 1 to n of n.

    For inputs 1, 2, 3, 4, it should return 1, 3, 6, 10:

    ```
    triangle.eval(1) = {triangle.eval(1)}
    triangle.eval(2) = {triangle.eval(2)}
    triangle.eval(3) = {triangle.eval(3)}
    triangle.eval(4) = {triangle.eval(4)}
    ```
    """
    )
    return


@app.cell
def _(Constant, Func):
    triangle = Func(
        "triangle",
        ("n",),
        Constant(0)
    )
    return (triangle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dialects

        Dialects are namespaces that group together related runtime and compile-time constructs.
        The most widely used dialect in xDSL and MLIR is `builtin`, which mostly comprises definitions of useful types and .
        For this tutorial, we'll be using constructs defined in the `builtin` for compile-time values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Attributes""")
    return


@app.cell
def _(mo):
    from xdsl.dialects.builtin import IntAttr, IntegerAttr, IntegerType, i32

    mo.md(r"""
    ### `IntAttr`

    Attributes store compile-time information such as constants and types.

    The first attribute we'll need is `IntAttr`, which wraps a Python integer:

    ```python
    IntAttr(1)
    ```

    Python integers are infinite-width, which makes them nice to use, but not appropriate if we want to represent values that will fit into a register on our hardware.
    To represent fixed-width integers we need to leverage two more attributes, `IntegerType` and `IntegerAttr`.
    """
    )
    return IntAttr, IntegerAttr, IntegerType, i32


@app.cell
def _(mo):
    mo.md(
        r"""
        ### IntegerType

        `IntegerType` encodes two properties of an integer type, its bitwidth and signedness.

        Signedness is handled in a relatively unusual way in MLIR and xDSL.
        While they allow the type to represent that an integer is signed or unsigned, like in many programming languages, they also allow the user not to specify the signedness of the value.
        In practice, it means that an 8-bit integer with bits 0xAB could either represent the value 171 (unsigned), -85 (signed), or either.
        For the latter, the convention is for computation
        """
    )
    return


@app.cell
def _(i32):
    repr(i32)
    return


if __name__ == "__main__":
    app.run()
