import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from sympy import S, symbols, Expr, Add, Mul, Pow, Sum, Integer
    from sympy.core.symbol import Symbol

    from xdsl.ir import Operation, SSAValue, Region, Block
    from xdsl.dialects.builtin import ModuleOp, Float64Type, FloatAttr, IntegerType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.arith import AddfOp, MulfOp, ConstantOp
    from xdsl.dialects.scf import ForOp, YieldOp
    from xdsl.dialects.experimental.math import PowFOp, SqrtOp
    from xdsl.builder import Builder, InsertPoint
    return (
        Add,
        AddfOp,
        Block,
        Builder,
        ConstantOp,
        Expr,
        Float64Type,
        FloatAttr,
        ForOp,
        FuncOp,
        InsertPoint,
        Integer,
        IntegerType,
        ModuleOp,
        Mul,
        MulfOp,
        Operation,
        Pow,
        PowFOp,
        Region,
        ReturnOp,
        S,
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
def _(Sum, symbols):
    x, y = symbols("x y")
    expr = x * y + y ** 2
    expr2 = Sum(x ** 2, (x, 1, 5))
    return expr, expr2, x, y


@app.cell
def _(
    Add,
    AddfOp,
    Block,
    Builder,
    ConstantOp,
    Expr,
    Float64Type,
    FloatAttr,
    ForOp,
    InsertPoint,
    Integer,
    IntegerType,
    Mul,
    MulfOp,
    Pow,
    PowFOp,
    Region,
    SSAValue,
    Sum,
    Symbol,
    YieldOp,
):
    def emit_op(expr: Expr, builder: Builder, args: dict[str, SSAValue]):
        if isinstance(expr, Symbol):
            return args[expr.name]

        if isinstance(expr, Integer):
            constant = ConstantOp(FloatAttr(float(expr), Float64Type()))
            builder.insert(constant)
            return constant.result

        if expr.func == Add:
            lhs = emit_op(expr.args[0], builder, args)
            rhs = emit_op(expr.args[1], builder, args)
            add_op = AddfOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Mul:
            lhs = emit_op(expr.args[0], builder, args)
            rhs = emit_op(expr.args[1], builder, args)
            add_op = MulfOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Pow:
            lhs = emit_op(expr.args[0], builder, args)
            rhs = emit_op(expr.args[1], builder, args)
            add_op = PowFOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Sum:
            lower_bound = emit_op(expr.args[1][1], builder, args)
            upper_bound = emit_op(expr.args[1][2], builder, args)
            one = ConstantOp(0, IntegerType(64))
            zero = ConstantOp(FloatAttr(float(expr), Float64Type()))
            for_op = ForOp(lower_bound, upper_bound, one, [zero], Region(Block(arg_types=[IntegerType(64)])))
            builder.insert(for_op)
            old_insert_point = builder.insertion_point
            builder.insertion_point = InsertPoint.at_end(for_op.body.block)
            res = emit_op(expr.args[0], builder, {**args, expr.args[1][0].name: for_op.body.block.args[0]})
            builder.insert(YieldOp(res))
            builder.insertion_point = old_insert_point

            return for_op.res[0]


        raise ValueError(f"No IR emitter for {expr.func}")
    return (emit_op,)


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
    expr,
    expr2,
):
    def emit_ir(expr: Expr) -> ModuleOp:
        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        arg_names = [arg.name for arg in expr.free_symbols]

        func = FuncOp("main", ([Float64Type()] * len(arg_names), [Float64Type()]))
        builder.insert(func)

        arg_values = {arg: value for arg, value in zip(arg_names, func.args)}
        for arg, value in arg_values.items():
            value.name_hint = arg

        builder.insertion_point = InsertPoint.at_end(func.body.block)
        result = emit_op(expr, builder, arg_values)

        builder.insert(ReturnOp(result))
        return module

    print(emit_ir(expr))
    print(emit_ir(expr2))
    return (emit_ir,)


if __name__ == "__main__":
    app.run()
