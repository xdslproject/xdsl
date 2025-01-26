# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sympy",
#     "xdsl",
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from sympy import S, symbols, Expr, Add, Mul, Sum, Integer, Float, E, I, re, im, Abs, Pow, Rational, Function, UnevaluatedExpr
    from sympy.core.symbol import Symbol

    from xdsl.ir import Operation, SSAValue, Region, Block, ParametrizedAttribute
    from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern, PatternRewriteWalker, GreedyRewritePatternApplier
    from xdsl.transforms.dead_code_elimination import region_dce
    from xdsl.traits import Pure
    from xdsl.irdl import irdl_op_definition, traits_def, IRDLOperation, irdl_attr_definition, operand_def, result_def
    from xdsl.dialects.builtin import ModuleOp, Float64Type, FloatAttr, IntegerType, IntegerAttr
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.arith import AddfOp, SubfOp, MulfOp, ConstantOp, AddiOp, MuliOp, SIToFPOp, FloatingPointLikeBinaryOperation, DivfOp
    from xdsl.dialects.scf import ForOp, YieldOp
    from xdsl.dialects.math import PowFOp, SqrtOp
    from xdsl.builder import Builder, InsertPoint
    return (
        Abs,
        Add,
        AddfOp,
        AddiOp,
        Block,
        Builder,
        ConstantOp,
        DivfOp,
        E,
        Expr,
        Float,
        Float64Type,
        FloatAttr,
        FloatingPointLikeBinaryOperation,
        ForOp,
        FuncOp,
        Function,
        GreedyRewritePatternApplier,
        I,
        IRDLOperation,
        InsertPoint,
        Integer,
        IntegerAttr,
        IntegerType,
        ModuleOp,
        Mul,
        MulfOp,
        MuliOp,
        Operation,
        ParametrizedAttribute,
        PatternRewriteWalker,
        PatternRewriter,
        Pow,
        PowFOp,
        Pure,
        Rational,
        Region,
        ReturnOp,
        RewritePattern,
        S,
        SIToFPOp,
        SSAValue,
        SqrtOp,
        SubfOp,
        Sum,
        Symbol,
        UnevaluatedExpr,
        YieldOp,
        im,
        irdl_attr_definition,
        irdl_op_definition,
        op_type_rewrite_pattern,
        operand_def,
        re,
        region_dce,
        result_def,
        symbols,
        traits_def,
    )


@app.cell(hide_code=True)
def _(FloatingPointLikeBinaryOperation, irdl_op_definition):
    # The Pow function, currently missing from xdsl arith dialect:

    @irdl_op_definition
    class PowfOp(FloatingPointLikeBinaryOperation):
        name = "arith.powf"
    return (PowfOp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Writing an IR emitter for SymPy

        SymPy is a Python library for symbolic mathematics. The goal here is to convert SymPy
        expressions to MLIR core dialects, namely `builtin`, `func`, and `arith`. In this exercise, we will only handle a subset of SymPy that only deals with real numbers for simplicity.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Some SymPy examples""")
    return


@app.cell
def _(symbols):
    # Define variables that can be used in the rest of the notebook
    # x, y, z, t are non-zere reals, while a, b, c, d are non-zero integers
    x, y, z, t = symbols("x y z t", real=True, zero=False)
    a, b, c, d = symbols("a b c d", integer=True, zero=False)

    # SymPy uses overloaded Python operators to define expressions.
    # Expressions are automatically simplified by SymPy.
    print('"x + y * z" -> ', x + y * z)
    print('"x + x" -> ', x + x)
    print('"x - x" -> ', x - x)
    return a, b, c, d, t, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expected MLIR output

        Here is an example of SymPy code, and its expected MLIR output.
        We want to only output code from the `arith` dialect, which deals with arithmetic.

        For this exercise, we will only use the following operations: arith.constant, arith.addf, arith.mulf, arith.powf. Their MLIR documentation can be found [here](https://mlir.llvm.org/docs/Dialects/ArithOps/).

        Here is the expected MLIR output for the expression `x + x`:
        ```
        builtin.module {
          func.func @main(%x : f64) -> f64 {
            %add = arith.addf %x, %x : f64
            func.return %add : f64
          }
        }
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introspecting SymPy expressions

        SymPy expressions all inherit from `Expr`. For all our examples, `Expr` are a tree of expressions, where nodes are function calls and leaves are either symbols or constants.

        * Function calls can be introspected using the `func` and `args` fields, which respectively return the associated function and its arguments (its node children). Additionally, the type of a function call is equal to the value returned by `func`.
        * Constants can be converted to floats using the `float` function.
        * Each symbol is a function argument.

        Here are some examples of SymPy introspection:
        """
    )
    return


@app.cell
def _(x):
    # A SymPy expression
    expression = x + x

    print("expression:", expression)

    # The func argument is equal to the type of the AST node
    print("type:", type(expression))
    print("func:", expression.func)

    # The operation argument and their types:
    print("args:", expression.args)
    print("type(args[0]):", type(expression.args[0]))
    print("type(args[1]):", type(expression.args[1]))
    return (expression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise: Emitting MLIR IR from SymPy

        Your goal is to convert SymPy AST nodes to MLIR IR, using xDSL.
        We are giving you most of the boilerplate, so you only have to focus on emitting IR for each SymPy AST node.

        The following function will print an expression, and print the resulting MLIR that you are emitting:
        """
    )
    return


@app.cell
def _(Expr, emit_ir):
    def print_ir(expr: Expr):
        # Print the SymPy expression
        print(expr)

        # Converts the SymPy expression to an MLIR `builtin.module` operation
        try:
            op = emit_ir(expr)

            # Check that the operation verifies, and prints the operation
            op.verify()
            print(op)
        except Exception as e:
            print("Error while converting expression: ", e)

        # Print a separator
        print("\n\n")
    return (print_ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This function takes a SymPy expression, creates a module and a function, and calls the main recursive function to convert SymPy AST.""")
    return


@app.cell
def _(Attribute, Expr, Float64Type, IntegerType):
    # Get the MLIR type for a SymPy expression
    def get_type(expr: Expr) -> Attribute:
        if expr.is_integer:
            return IntegerType(64)
        elif expr.is_extended_real:
            return Float64Type()
        else:
            raise Exception(f"Unknown MLIR type for expression {expr}. Please make sure there cannot be a division by zero, or a power of a negative value.")
    return (get_type,)


@app.cell
def _(
    Builder,
    Expr,
    FuncOp,
    InsertPoint,
    ModuleOp,
    ReturnOp,
    emit_op,
    get_type,
):
    def emit_ir(expr: Expr) -> ModuleOp:
        # Create a module, and create a builder at the beginning of its only block
        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Create the MLIR types for each symbol.
        arg_types = [get_type(arg) for arg in expr.free_symbols]

        # Create a new function and inserts it inside the module.
        func = FuncOp("main", (arg_types, [get_type(expr)]))
        builder.insert(func)

        # Associate each symbol with its MLIR name.
        arg_values = {arg: value for arg, value in zip(expr.free_symbols, func.args)}

        # Set the name for each function argument. This is only to get better names
        # for IR values.
        for arg, value in arg_values.items():
            value.name_hint = arg.name

        # Set the builder insertion point inside the function.
        builder.insertion_point = InsertPoint.at_end(func.body.block)

        # Convert the expression into MLIR IR inside the function.
        result = emit_op(expr, builder, arg_values)

        # Insert a return statement at the end of the function.
        builder.insert(ReturnOp(result))
        return module
    return (emit_ir,)


@app.cell
def _(mo):
    mo.md(r"""Finally, here are the functions that you should complete. `emit_op` is fully complete, and emits the necessary IR for a SymPy expression. `emit_integer_op` and `emit_float_op` emits the operations for integer and float operations, and are only partially implemented.""")
    return


@app.cell
def _(
    Builder,
    Expr,
    Float,
    Float64Type,
    Integer,
    IntegerType,
    SIToFPOp,
    SSAValue,
    Symbol,
    get_type,
):
    def emit_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ):
        type = get_type(expr)
        if isinstance(type, IntegerType):
            return emit_integer_op(expr, builder, args)
        elif isinstance(type, Float64Type):
            return emit_real_op(expr, builder, args)
        else:
            raise Exception("Unknown function to emit IR for MLIR type ", type)

    def emit_integer_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ):
        # Handle symbolic values
        if isinstance(expr, Symbol):
            return args[expr]

        # Handle constants
        if isinstance(expr, Integer):
            # int(expr) returns the value of the `expr` constant
            raise NotImplementedError("Constants are not implemented")

        raise NotImplementedError(f"No IR emitter for integer function {expr.func}")

    def emit_real_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ):
        # If the expression is an integer expression, emits it and then convert it
        # back to a float expression.
        if expr.is_integer:
            res = emit_integer_op(expr, builder, args)
            op = builder.insert(SIToFPOp(res))
            return op.result

        # Handle constants
        if isinstance(expr, Float):
            # float(expr) returns the value of the `expr` constant
            raise NotImplementedError("Constants are not implemented")

        # Handle symbolic values
        if isinstance(expr, Symbol):
            return args[expr]

        raise NotImplementedError(f"No IR emitter for float function {expr.func}")
    return emit_integer_op, emit_op, emit_real_op


@app.cell
def _(mo):
    mo.md("""Here are a few simple examples that you should support first. For each test, the expression is printed, then either the MLIR code, or an error. Each of the operators used in these tests should only be converted to a single MLIR operation.""")
    return


@app.cell
def _(Float, Integer, a, b, print_ir, x, y):
    print_ir(Float(2))
    print_ir(Integer(2))

    print_ir(a + b)
    print_ir(a + b + x)
    print_ir(x + 2)
    print_ir(x * y)
    print_ir(a * b + x)
    print_ir(x + x)
    print_ir(x / y)
    print_ir(x - y)
    return


@app.cell
def _(mo):
    mo.md(r"""The following expression requires to handle the AST node `Abs`. Instead of converting it to `math.absf` operation, we taks you to write it using the formula `x < 0 ? -x : x` using only `arith` operations.""")
    return


@app.cell
def _(Abs, print_ir, x, y):
    print_ir(Abs(x + y))
    print_ir((x ** 2) ** y)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Supporting operations with regions

        Your next task is to handle operations that may have regions.

        As a first step, rewrite the lowering to `Abs` to output an `scf.if` instead of an `arith.select`. Then, as an harder task, support the `Sum` operation using an `scf.for` loop.

        Here are a few examples:
        """
    )
    return


@app.cell
def _(Abs, Sum, UnevaluatedExpr, a, b, c, print_ir, x, y):
    print_ir(Abs(x + y))
    print_ir((x ** 2) ** y)

    # The sum of all numbers from 0 to 10 (excluded)
    print_ir(Sum(x, (x, 0, 10)))

    # The triangle sum from 0 to a (excluded)
    print_ir(Sum(x*x, (x, 0, a)))

    # The computation of:
    # for b in range(0, a):
    #   for c in range(0, b):
    #      result += 1
    # We use an UnevaluatedExpr so that SymPy doesn't combine both sums
    print_ir(Sum(UnevaluatedExpr(Sum(1, (c, 0, b))), (b, 0, a)))
    return


if __name__ == "__main__":
    app.run()
