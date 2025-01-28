# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xdsl==0.27.0",
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from sympy import (
        S,
        symbols,
        Expr,
        Add,
        Mul,
        Sum,
        Integer,
        Float,
        E,
        I,
        re,
        im,
        Abs,
        Pow,
        Rational,
        Function,
        UnevaluatedExpr,
    )
    from sympy.core.symbol import Symbol

    from xdsl.ir import Operation, SSAValue, Region, Block, ParametrizedAttribute
    from xdsl.pattern_rewriter import (
        PatternRewriter,
        RewritePattern,
        op_type_rewrite_pattern,
        PatternRewriteWalker,
        GreedyRewritePatternApplier,
    )
    from xdsl.transforms.dead_code_elimination import region_dce
    from xdsl.traits import Pure
    from xdsl.irdl import (
        irdl_op_definition,
        traits_def,
        IRDLOperation,
        irdl_attr_definition,
        operand_def,
        result_def,
    )
    from xdsl.dialects.builtin import (
        ModuleOp,
        Float64Type,
        FloatAttr,
        IntegerType,
        IntegerAttr,
    )
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.arith import (
        AddfOp,
        SubfOp,
        MulfOp,
        ConstantOp,
        AddiOp,
        MuliOp,
        SIToFPOp,
        FloatingPointLikeBinaryOperation,
        DivfOp,
        SelectOp,
        CmpfOp,
    )
    from xdsl.dialects.scf import ForOp, YieldOp, IfOp
    from xdsl.dialects.math import PowFOp, SqrtOp
    from xdsl.builder import Builder, InsertPoint
    return (
        Abs,
        Add,
        AddfOp,
        AddiOp,
        Block,
        Builder,
        CmpfOp,
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
        IfOp,
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
        SelectOp,
        SqrtOp,
        SubfOp,
        Sum,
        Symbol,
        UnevaluatedExpr,
        YieldOp,
        im,
        irdl_attr_definition,
        irdl_op_definition,
        mo,
        op_type_rewrite_pattern,
        operand_def,
        re,
        region_dce,
        result_def,
        symbols,
        traits_def,
    )


@app.cell
def _(symbols):
    # Define variables that can be used in the rest of the notebook
    # x, y, z, t are non-zere reals, while a, b, c, d are non-zero integers
    x, y, z, t = symbols("x y z t", real=True, zero=False)
    a, b, c, d = symbols("a b c d", integer=True, zero=False)
    return a, b, c, d, t, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Solutions""")
    return


@app.cell(hide_code=True)
def _(Expr, emit_ir):
    def print_ir(expr: Expr):
        # Print the SymPy expression
        print(expr)

        # Converts the SymPy expression to an MLIR `builtin.module` operation
        op = emit_ir(expr)

        # Check that the operation verifies, and prints the operation
        op.verify()
        print(op)

        # Print a separator
        print("\n\n")
    return (print_ir,)


@app.cell(hide_code=True)
def _(Attribute, Expr, Float64Type, IntegerType):
    # Get the MLIR type for a SymPy expression
    def get_mlir_type(expr: Expr) -> Attribute:
        if expr.is_integer:
            return IntegerType(64)
        elif expr.is_extended_real:
            return Float64Type()
        else:
            raise Exception(
                f"Unknown MLIR type for expression {expr}. Please make sure there cannot be a division by zero, or a power of a negative value."
            )
    return (get_mlir_type,)


@app.cell(hide_code=True)
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
def _(
    Builder,
    Expr,
    Float64Type,
    IntegerType,
    SSAValue,
    Symbol,
    emit_integer_op,
    emit_real_op,
    get_mlir_type,
):
    def emit_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ):
        type = get_mlir_type(expr)
        if isinstance(type, IntegerType):
            return emit_integer_op(expr, builder, args)
        elif isinstance(type, Float64Type):
            return emit_real_op(expr, builder, args)
        else:
            raise Exception("Unknown function to emit IR for MLIR type ", type)
    return (emit_op,)


@app.cell
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

        # Hint: Implement here support for Add, Mul, and Pow (and later Abs and Sum)

        raise NotImplementedError(f"No IR emitter for float function {expr.func}")
    return emit_integer_op, emit_real_op


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
    print_ir(x**4)
    return


@app.cell
def _(Abs, print_ir, x, y):
    print_ir(Abs(x + y))
    print_ir((x**2) ** y)
    return


@app.cell
def _(Sum, a, b, print_ir):
    # The sum of all numbers from 0 to 10 (excluded)
    # You can access the Sum arguments with `args[0]`, and `args[1][0]`, `args[1][1]` and `args[1][2]`.
    print_ir(Sum(a, (a, 0, 10)))

    # The triangle sum from 0 to a (excluded)
    print_ir(Sum(b + b, (b, 0, a)))
    return


if __name__ == "__main__":
    app.run()
