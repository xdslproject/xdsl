# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "sympy==1.13.3",
#     "xdsl==0.27.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from sympy import S, symbols, Expr, Add, Mul, Sum, Integer, Float, E, I, re, im, Abs, Pow, Rational, Function, UnevaluatedExpr
    from sympy.core.symbol import Symbol

    from xdsl.ir import Attribute, Operation, SSAValue, Region, Block, ParametrizedAttribute
    from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern, PatternRewriteWalker, GreedyRewritePatternApplier
    from xdsl.transforms.dead_code_elimination import region_dce
    from xdsl.traits import Pure
    from xdsl.irdl import irdl_op_definition, traits_def, IRDLOperation, irdl_attr_definition, operand_def, result_def
    from xdsl.dialects.builtin import ModuleOp, Float64Type, FloatAttr, IntegerType, IntegerAttr
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.arith import AddfOp, SubfOp, MulfOp, ConstantOp, AddiOp, MuliOp, SIToFPOp, FloatingPointLikeBinaryOperation, DivfOp, SelectOp, CmpfOp
    from xdsl.dialects.scf import ForOp, YieldOp, IfOp
    from xdsl.dialects.math import PowFOp, SqrtOp
    from xdsl.builder import Builder, InsertPoint
    return (
        Abs,
        Add,
        AddfOp,
        AddiOp,
        Attribute,
        Block,
        Builder,
        CmpfOp,
        ConstantOp,
        Expr,
        Float,
        Float64Type,
        FloatAttr,
        ForOp,
        FuncOp,
        IfOp,
        InsertPoint,
        Integer,
        IntegerAttr,
        IntegerType,
        ModuleOp,
        Mul,
        MulfOp,
        MuliOp,
        Pow,
        PowFOp,
        Region,
        ReturnOp,
        SIToFPOp,
        SSAValue,
        SelectOp,
        SubfOp,
        Sum,
        Symbol,
        YieldOp,
        mo,
        symbols,
    )


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
    return a, b, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Expected MLIR output

    We want to only output code from the `arith` dialect (and `math.powf`), which deals with arithmetic.

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
        """
    The operations you will have to use have the following constructors.:

    * arith.constant: ConstantOp(IntegerAttr(int_cst, IntegerType(64)))
    * arith.constant: ConstantOp(FloatAttr(float_cst, Float64Type()))
    * arith.addi: AddiOp(lhs, rhs)
    * arith.muli: MuliOp(lhs, rhs)
    * arith.addf: AddfOp(lhs, rhs)
    * arith.mulf: MulfOp(lhs, rhs)
    * math.powf: PowFOp(lhs, rhs)
    * arith.select: SelectOp(cond, lhs, rhs)
        * Represents the formula `if cond then lhs else rhs`
    * arith.cmpf: CmpfOp("olt", lhs, rhs)
        * Represents `lhs < rhs` with floating points
    * scf.if: IfOp(cond, region1, region2)
        * Regions should have a single block, without block arguments. The last operation in the regions should be an `scf.yield` with constructor `YieldOp([result])`
        * `.results[0]` is used to get the result of `IfOp`
    * scf.for: ForOp(lower_bound, upper_bound, step, [first_acc], region)
        * `first_acc` is the initial value of the accumulator
        * `region` should have a single block with two block arguments. One for the value we iterate on (we use `i64`), and the second for the accumulator (we use `f64` here).
        * `.results[0]` is used to get the result of `ForOp`
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
    return


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

        try:
            # Converts the SymPy expression to an MLIR `builtin.module` operation
            op = emit_ir(expr)
        except NotImplementedError as e:
            print(e, "\n\n")
            return

        # Check that the operation verifies, and prints the operation
        op.verify()
        print(op)

        # Print a separator
        print("\n\n")
    return (print_ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    The following function returns the MLIR type of the expression result. As we are only handling integer and real types, we only returns either the `i64` or `f64` types, which correspond to 64bits integers and floating points.

    For instance, the type of `x + y` is `f64`, as `x` and `y` are floating points. The type of `a * b` is `i64`, as both `a` and `b` are integers.
    """
    )
    return


@app.cell
def _(Attribute, Expr, Float64Type, IntegerType):
    # Get the MLIR type for a SymPy expression
    def get_mlir_type(expr: Expr) -> Attribute:
        if expr.is_integer:
            return IntegerType(64)
        elif expr.is_extended_real:
            return Float64Type()
        else:
            raise Exception(f"Unknown MLIR type for expression {expr}. Please make sure there cannot be a division by zero, or a power of a negative value.")
    return (get_mlir_type,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The following function (`emit_ir`) should be called to emit MLIR IR from a SymPy expression. It takes a SymPy expression, creates a module and a function, and starts the recursion on the SymPy AST to emit MLIR IR.""")
    return


@app.cell
def _(
    Builder,
    Expr,
    FuncOp,
    InsertPoint,
    ModuleOp,
    ReturnOp,
    emit_op,
    get_mlir_type,
):
    def emit_ir(expr: Expr) -> ModuleOp:
        # Create a module, and create a builder at the beginning of its only block
        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Create the MLIR types for each symbol.
        arg_types = [get_mlir_type(arg) for arg in expr.free_symbols]

        # Create a new function and inserts it inside the module.
        func = FuncOp("main", (arg_types, [get_mlir_type(expr)]))
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


@app.cell(hide_code=True)
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
    get_mlir_type,
):
    def emit_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ) -> SSAValue:
        type = get_mlir_type(expr)
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
    ) -> SSAValue:
        # Handle symbolic values
        if isinstance(expr, Symbol):
            return args[expr]

        # Handle constants
        if isinstance(expr, Integer):
            # Hint: int(expr) returns the value of the `expr` constant
            raise NotImplementedError("Integer constants are not implemented")

        # Hint: Implement here support for Add and Mul

        raise NotImplementedError(f"No IR emitter for integer function {expr.func}")

    def emit_real_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ) -> SSAValue:
        # If the expression is an integer expression, emits it and then convert it
        # back to a float expression.
        if expr.is_integer:
            res = emit_integer_op(expr, builder, args)
            op = builder.insert(SIToFPOp(res, Float64Type()))
            return op.result

        # Handle constants
        if isinstance(expr, Float):
            # Hint: float(expr) returns the value of the `expr` constant
            raise NotImplementedError("Float constants are not implemented")

        # Handle symbolic values
        if isinstance(expr, Symbol):
            return args[expr]

        # Hint: Implement here support for Add, Mul, and Pow (and later Abs and Sum)

        raise NotImplementedError(f"No IR emitter for float function {expr.func}")
    return (emit_op,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Here are a few simple examples that you should support first. For each test, the expression is printed, then either the MLIR code, or an error. Each of the operators used in these tests should only be converted to a single MLIR operation.""")
    return


@app.cell
def _(Float, Integer, a, b, print_ir, x, y):
    print_ir(Float(2))
    print_ir(Integer(2))

    # Adds two integers
    print_ir(a + b)

    # Adds two integers with a real
    print_ir(a + b + x)

    # Multiplies two reals
    print_ir(x * y)

    # Multiplies three reals
    print_ir(a * b + x)

    # Add two reals
    print_ir(x + x)

    # Square a real
    print_ir(x ** 4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The following expression requires to handle the AST node `Abs`. Instead of converting it to `math.absf` operation, we taks you to write it using the formula `x < 0 ? 0-x : x` using only `arith` operations. Hint, you should use `arith.select` for expressing the conditional.""")
    return


@app.cell
def _(Abs, print_ir, x, y):
    print_ir(Abs(x + y))
    print_ir((x ** 2) ** y)
    return


@app.cell(hide_code=True)
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
def _(Abs, Sum, a, b, print_ir, x, y):
    print_ir(Abs(x + y))
    print_ir((x ** 2) ** y)

    # The sum of all numbers from 0 to 10 (excluded)
    # You can access the Sum arguments with `args[0]`, and `args[1][0]`, `args[1][1]` and `args[1][2]`.
    print_ir(Sum(a, (a, 0, 10)))

    # The triangle sum from 0 to a (excluded)
    print_ir(Sum(b+b, (b, 0, a)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Solutions

    Here is a solution for the implementation of `emit_real_op` and `emit_integer_op`.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    Abs,
    Add,
    AddfOp,
    AddiOp,
    Block,
    Builder,
    CmpfOp,
    ConstantOp,
    Expr,
    Float,
    Float64Type,
    FloatAttr,
    ForOp,
    IfOp,
    InsertPoint,
    Integer,
    IntegerAttr,
    IntegerType,
    Mul,
    MulfOp,
    MuliOp,
    Pow,
    PowFOp,
    Region,
    SIToFPOp,
    SSAValue,
    SelectOp,
    SubfOp,
    Sum,
    Symbol,
    YieldOp,
):
    def solution():
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
                constant_op = builder.insert(
                    ConstantOp(IntegerAttr(int(expr), IntegerType(64)))
                )
                return constant_op.result

            if isinstance(expr, Add):
                lhs = emit_integer_op(expr.args[0], builder, args)
                rhs = emit_integer_op(expr.args[1], builder, args)
                add_op = builder.insert(AddiOp(lhs, rhs))
                return add_op.result

            if isinstance(expr, Mul):
                lhs = emit_integer_op(expr.args[0], builder, args)
                rhs = emit_integer_op(expr.args[1], builder, args)
                add_op = builder.insert(MuliOp(lhs, rhs))
                return add_op.result

            # Hint: Implement here support for Add and Mul

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
                op = builder.insert(SIToFPOp(res, Float64Type()))
                return op.result

            # Handle constants
            if isinstance(expr, Float):
                constant_op = builder.insert(
                    ConstantOp(FloatAttr(float(expr), Float64Type()))
                )
                return constant_op

            # Handle symbolic values
            if isinstance(expr, Symbol):
                return args[expr]

            if isinstance(expr, Add):
                lhs = emit_real_op(expr.args[0], builder, args)
                rhs = emit_real_op(expr.args[1], builder, args)
                add_op = builder.insert(AddfOp(lhs, rhs))
                return add_op.result

            if isinstance(expr, Mul):
                lhs = emit_real_op(expr.args[0], builder, args)
                rhs = emit_real_op(expr.args[1], builder, args)
                add_op = builder.insert(MulfOp(lhs, rhs))
                return add_op.result

            if isinstance(expr, Pow):
                lhs = emit_real_op(expr.args[0], builder, args)
                rhs = emit_real_op(expr.args[1], builder, args)
                add_op = builder.insert(PowFOp(lhs, rhs))
                return add_op.result

            if isinstance(expr, Abs):
                # The arith.select solution
                if False:
                    arg = emit_real_op(expr.args[0], builder, args)
                    zero = builder.insert(ConstantOp(FloatAttr(0, Float64Type()))).result
                    neg = builder.insert(SubfOp(zero, arg)).result
                    is_neg = builder.insert(CmpfOp(arg, zero, "olt")).result
                    select = builder.insert(SelectOp(is_neg, neg, arg)).result
                    return select

                # The scf.if solution
                arg = emit_real_op(expr.args[0], builder, args)
                zero = builder.insert(ConstantOp(FloatAttr(0, Float64Type()))).result
                is_neg = builder.insert(CmpfOp(arg, zero, "olt")).result

                lhs_region = Region([Block()])
                builder2 = Builder(InsertPoint.at_end(lhs_region.block))
                neg = builder2.insert(SubfOp(zero, arg)).result
                builder2.insert(YieldOp(neg))

                rhs_region = Region([Block()])
                builder3 = Builder(InsertPoint.at_end(rhs_region.block))
                builder3.insert(YieldOp(neg))

                if_res = builder.insert(
                    IfOp(is_neg, Float64Type(), lhs_region, rhs_region)
                ).results[0]

                return if_res

            if isinstance(expr, Sum):
                zero = builder.insert(ConstantOp(FloatAttr(0, Float64Type()))).result
                lb = emit_integer_op(expr.args[1][1], builder, args)
                ub = emit_integer_op(expr.args[1][2], builder, args)
                step = builder.insert(ConstantOp(IntegerAttr(0, IntegerType(64))))
                region = Region([Block(arg_types=[IntegerType(64), Float64Type()])])
                accumulator = region.block.args[1]

                b2 = Builder(InsertPoint.at_end(region.block))
                arg = emit_real_op(
                    expr.args[0], b2, args | {expr.args[1][0]: region.block.args[0]}
                )
                add = b2.insert(AddfOp(arg, accumulator)).result
                b2.insert(YieldOp(add))

                return builder.insert(ForOp(lb, ub, step, [zero], region)).results[0]

            raise NotImplementedError(f"No IR emitter for float function {expr.func}")
    return


if __name__ == "__main__":
    app.run()
