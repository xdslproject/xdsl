import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from sympy import S, symbols, Expr, Add, Mul, Pow, Sum, Integer, Float
    from sympy.core.symbol import Symbol

    from xdsl.ir import Operation, SSAValue, Region, Block
    from xdsl.dialects.builtin import ModuleOp, Float64Type, FloatAttr, IntegerType, IntegerAttr
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.arith import AddfOp, MulfOp, ConstantOp, AddiOp, MuliOp, SIToFPOp
    from xdsl.dialects.scf import ForOp, YieldOp
    from xdsl.dialects.experimental.math import PowFOp, SqrtOp
    from xdsl.builder import Builder, InsertPoint
    return (
        Add,
        AddfOp,
        AddiOp,
        Block,
        Builder,
        ConstantOp,
        Expr,
        Float,
        Float64Type,
        FloatAttr,
        ForOp,
        FuncOp,
        InsertPoint,
        Integer,
        IntegerAttr,
        IntegerType,
        ModuleOp,
        Mul,
        MulfOp,
        MuliOp,
        Operation,
        Pow,
        PowFOp,
        Region,
        ReturnOp,
        S,
        SIToFPOp,
        SSAValue,
        SqrtOp,
        Sum,
        Symbol,
        YieldOp,
        mo,
        symbols,
    )


@app.cell
def _(mo):
    mo.md(r"""# Emitting MLIR IR from Sympy""")
    return


@app.cell
def _(
    Add,
    AddfOp,
    AddiOp,
    Block,
    Builder,
    ConstantOp,
    Expr,
    Float,
    Float64Type,
    FloatAttr,
    ForOp,
    InsertPoint,
    Integer,
    IntegerAttr,
    IntegerType,
    Mul,
    MulfOp,
    MuliOp,
    Region,
    SIToFPOp,
    SSAValue,
    Sum,
    Symbol,
    YieldOp,
):
    def get_type(expr: Expr) -> IntegerType | Float64Type:
        return IntegerType(64) if expr.is_integer else Float64Type()

    def emit_op(expr: Expr, builder: Builder, args: dict[str, SSAValue], expected_type: IntegerType | Float64Type):
        # Handle conversions from integer to float
        if isinstance(expected_type, Float64Type) and isinstance(get_type(expr), IntegerType):
            res = emit_op(expr, builder, args, IntegerType(64))
            convert_op = SIToFPOp(res, Float64Type())
            builder.insert(convert_op)
            return convert_op.result

        if expected_type != get_type(expr):
            raise ValueError("Wrong typing")

        # Handle symbolic values
        if isinstance(expr, Symbol):
            value = args[expr.name]

            # Convert values to the right type if necessary
            if isinstance(expected_type, Float64Type) and isinstance(get_type(expr), IntegerType):
                convert_op = SIToFPOp(res, Float64Type())
                builder.insert(convert_op)
                value = convert_op.result
            return value

        if isinstance(expr, Integer):
            constant = ConstantOp(IntegerAttr(int(expr), IntegerType(64)))
            builder.insert(constant)
            return constant.result

        if isinstance(expr, Float):
            constant = ConstantOp(FloatAttr(float(expr), Float64Type()))
            builder.insert(constant)
            return constant.result

        if expr.func == Add:
            lhs = emit_op(expr.args[0], builder, args, expected_type)
            rhs = emit_op(expr.args[1], builder, args, expected_type)
            if isinstance(expected_type, IntegerType):
                add_op = AddiOp(lhs, rhs)
            else:
                add_op = AddfOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Mul:
            lhs = emit_op(expr.args[0], builder, args, expected_type)
            rhs = emit_op(expr.args[1], builder, args, expected_type)
            if isinstance(expected_type, IntegerType):
                add_op = MuliOp(lhs, rhs)
            else:
                add_op = MulfOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Sum:
            lower_bound = emit_op(expr.args[1][1], builder, args, IntegerType(64))
            upper_bound = emit_op(expr.args[1][2], builder, args, IntegerType(64))

            one = ConstantOp(IntegerAttr(1, IntegerType(64)))
            builder.insert(one)
            if isinstance(expected_type, Float64Type):
                zero = ConstantOp(FloatAttr(0, expected_type))
            else:
                zero = ConstantOp(IntegerAttr(0, expected_type))
            builder.insert(zero)

            for_op = ForOp(lower_bound, upper_bound, one, [zero.result], Region(Block(arg_types=[IntegerType(64), Float64Type()])))
            builder.insert(for_op)
            accumulator = for_op.body.block.args[1]

            old_insert_point = builder.insertion_point
            builder.insertion_point = InsertPoint.at_end(for_op.body.block)
            res = emit_op(expr.args[0], builder, {**args, expr.args[1][0].name: for_op.body.block.args[0]}, expected_type)
            if isinstance(expected_type, IntegerType):
                add = AddiOp(res, accumulator)
            else:
                add = AddfOp(res, accumulator)
            builder.insert(add)
            builder.insert(YieldOp(add.result))
            builder.insertion_point = old_insert_point

            return for_op.res[0]


        raise ValueError(f"No IR emitter for {expr.func}")
    return emit_op, get_type


@app.cell
def _(
    Builder,
    Expr,
    FuncOp,
    InsertPoint,
    ModuleOp,
    ReturnOp,
    Sum,
    emit_op,
    get_type,
    symbols,
):
    def emit_ir(expr: Expr) -> ModuleOp:
        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        arg_names = [arg.name for arg in expr.free_symbols]
        arg_types = [get_type(arg) for arg in expr.free_symbols]

        func = FuncOp("main", (arg_types, [get_type(expr)]))
        builder.insert(func)

        arg_values = {arg: value for arg, value in zip(arg_names, func.args)}
        for arg, value in arg_values.items():
            value.name_hint = arg

        builder.insertion_point = InsertPoint.at_end(func.body.block)
        expected_type = get_type(expr)
        result = emit_op(expr, builder, arg_values, expected_type)

        builder.insert(ReturnOp(result))
        return module

    def test(expr: Expr):
        print(expr)
        op = emit_ir(expr)
        op.verify()
        print(op)
        print("\n" * 3)

    x, y = symbols("x y", real=True)
    a, b = symbols("a b", integer=True)

    test(x * y + y)
    test(x - y)
    test(x + x)
    test(Sum(a * 2, (a, 1 + b, 5)))
    return a, b, emit_ir, test, x, y


if __name__ == "__main__":
    app.run()
