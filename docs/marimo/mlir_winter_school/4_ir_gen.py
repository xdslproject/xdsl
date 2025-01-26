import marimo

__generated_with = "0.10.17"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from sympy import S, symbols, Expr, Add, Mul, Sum, Integer, Float, E, I, re, im, Abs, Pow, Rational, Function
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
    from xdsl.dialects.experimental.math import PowFOp, SqrtOp
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
    mo.md(
        r"""
        ## Some SymPy examples

        """
    )
    return


@app.cell
def _(symbols):
    # Define 4 variables that can be used in the rest of the notebook
    x, y, z, t = symbols("x y z t", real=True)

    # SymPy uses overloaded Python operators to define expressions.
    # Expressions are automatically simplified by SymPy.
    print('"x + y * z" -> ', x + y * z)
    print('"x + x" -> ', x + x)
    print('"x - x" -> ', x - x)
    return t, x, y, z


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
        We are giving you most of the boilerplate, so you only have to focus on translating each operation.

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
        except Exception as e:
            print("Error while converting expression: ", e)
            return

        # Check that the operation verifies
        op.verify()

        # Print the operation
        print(op, "\n\n")
    return (print_ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This function takes a SymPy expression, creates a module and a function, and calls the main recursive function to convert SymPy AST.""")
    return


@app.cell
def _(
    Builder,
    Expr,
    Float64Type,
    FuncOp,
    InsertPoint,
    ModuleOp,
    ReturnOp,
    emit_op,
):
    def emit_ir(expr: Expr) -> ModuleOp:
        # Create a module, and create a builder at the beginning of its only block
        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Create the MLIR types for each symbol. We use `f64` here.
        arg_types = [Float64Type() for arg in expr.free_symbols]

        # Create a new function and inserts it inside the module.
        func = FuncOp("main", (arg_types, [Float64Type()]))
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
    mo.md(r"""Finally, here is the main function that you should complete. This function converts recursively each SymPy AST node into MLIR operations.""")
    return


@app.cell
def _(
    Add,
    AddfOp,
    Builder,
    ConstantOp,
    Expr,
    Float,
    Float64Type,
    FloatAttr,
    Integer,
    SSAValue,
    Symbol,
):
    def emit_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ):
        # Handle symbolic values
        if isinstance(expr, Symbol):
            # Just return the value associated to the symbol
            return args[expr]

        # Handle constants
        if isinstance(expr, Float) or isinstance(expr, Integer):
            constant = ConstantOp(FloatAttr(float(expr), Float64Type()))
            builder.insert(constant)
            return constant.result

        # Here is an example on how to convert a simple operation:
        if expr.func == Add:
            lhs = emit_op(expr.args[0], builder, args)
            rhs = emit_op(expr.args[1], builder, args)
            add_op = AddfOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        raise ValueError(f"No IR emitter for {expr.func}")
    return (emit_op,)


@app.cell
def _():
    return


@app.cell
def _(print_ir, x, y):
    print_ir(x + y)
    print_ir(x * y)
    print_ir(x + x)
    print_ir(x / y)
    print_ir(x - y)
    print_ir(x ** y)
    return


if __name__ == "__main__":
    app.run()
