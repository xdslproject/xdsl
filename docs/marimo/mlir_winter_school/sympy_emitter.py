import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from sympy import S, symbols, Expr, Add, Mul, Sum, Integer, Float, E, I, re, im, Abs, Pow, Rational
    from sympy.core.symbol import Symbol

    from xdsl.ir import Operation, SSAValue, Region, Block
    from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern, PatternRewriteWalker, GreedyRewritePatternApplier
    from xdsl.transforms.dead_code_elimination import region_dce
    from xdsl.traits import Pure
    from xdsl.irdl import irdl_op_definition, traits_def
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
        GreedyRewritePatternApplier,
        I,
        InsertPoint,
        Integer,
        IntegerAttr,
        IntegerType,
        ModuleOp,
        Mul,
        MulfOp,
        MuliOp,
        Operation,
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
        irdl_op_definition,
        mo,
        op_type_rewrite_pattern,
        re,
        region_dce,
        symbols,
        traits_def,
    )


@app.cell
def _(FloatingPointLikeBinaryOperation, irdl_op_definition):
    @irdl_op_definition
    class PowfOp(FloatingPointLikeBinaryOperation):
        name = "arith.powf"

    return (PowfOp,)


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
    Pow,
    PowfOp,
    Rational,
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

        if isinstance(expr, Rational):
            constant = ConstantOp(FloatAttr(expr.p / expr.q, Float64Type()))
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

        if expr.func == Pow:
            assert isinstance(expected_type, Float64Type)
            lhs = emit_op(expr.args[0], builder, args, expected_type)
            rhs = emit_op(expr.args[1], builder, args, expected_type)
            pow_op = PowfOp(lhs, rhs)
            builder.insert(pow_op)
            return pow_op.result

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
    Abs,
    Builder,
    Expr,
    FuncOp,
    I,
    InsertPoint,
    ModuleOp,
    ReturnOp,
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

    x, y, z, t = symbols("x y z t", real=True)
    a, b = symbols("a b", integer=True)

    # test(x * y + y)
    # test(x - y)
    # test(x + x)
    # test(Sum(a * 2, (a, 1 + b, 5)))
    test(Abs(x + y*I))
    test(Abs(x + y*I) * Abs(z + t*I))
    test(Abs((x + y*I) * (z + t*I)))
    return a, b, emit_ir, t, test, x, y, z


@app.cell
def _(mo):
    mo.md("""# Optimizations""")
    return


@app.cell
def _(
    Abs,
    AddfOp,
    ConstantOp,
    Expr,
    Float64Type,
    FloatAttr,
    GreedyRewritePatternApplier,
    I,
    MulfOp,
    Operation,
    PatternRewriteWalker,
    PatternRewriter,
    PowfOp,
    RewritePattern,
    SIToFPOp,
    SubfOp,
    emit_ir,
    region_dce,
    t,
    x,
    y,
    z,
):
    class SIToFPConstantPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, SIToFPOp):
                return
            if not isinstance(op.input.owner, ConstantOp):
                return
            new_op = ConstantOp(FloatAttr(op.input.owner.value.value.data, Float64Type()))
            rewriter.replace_op(op, new_op)

    class AddTimesMinusOnePattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, AddfOp):
                return
            if not isinstance(mul := op.rhs.owner, MulfOp):
                return
            if not isinstance(constant := mul.lhs.owner, ConstantOp):
                return
            if constant.value.value.data != -1.0:
                return
            rewriter.replace_op(op, SubfOp(op.lhs, mul.rhs))

    class Pow2Pattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, PowfOp):
                return
            if not isinstance(cst := op.rhs.owner, ConstantOp):
                return
            if cst.value.value.data != 2.0:
                return
            rewriter.replace_op(op, MulfOp(op.lhs, op.lhs))

    def test_with_opts(expr: Expr):
        print(expr)
        op = emit_ir(expr)
        op.verify()
        print("Before optimizations:", op)
        PatternRewriteWalker(GreedyRewritePatternApplier([SIToFPConstantPattern(), AddTimesMinusOnePattern(), Pow2Pattern()])).rewrite_module(op)
        region_dce(op.body)
        print("After optimizations:", op)
        print("\n" * 3)

    test_with_opts(x - y)
    test_with_opts(Abs(x + y*I))
    test_with_opts(Abs(x + y*I) * Abs(z + t*I))
    return (
        AddTimesMinusOnePattern,
        Pow2Pattern,
        SIToFPConstantPattern,
        test_with_opts,
    )


@app.cell
def _(mo):
    mo.md("""# New dialect for complex numbers""")
    return


if __name__ == "__main__":
    app.run()
