import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

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
    Float64Type,
    IRDLOperation,
    ParametrizedAttribute,
    SSAValue,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
):
    @irdl_attr_definition
    class ComplexType(ParametrizedAttribute):
        name = "complex.complex"

    @irdl_op_definition
    class CreateOp(IRDLOperation):
        name = "complex.create"

        re = operand_def(Float64Type())
        im = operand_def(Float64Type())

        result = result_def(ComplexType())

        def __init__(self, lhs: SSAValue, rhs: SSAValue):
            super().__init__(operands=[lhs, rhs], result_types=[ComplexType()])

    @irdl_op_definition
    class ReOp(IRDLOperation):
        name = "complex.re"

        arg = operand_def(ComplexType())

        result = result_def(Float64Type())

        def __init__(self, arg: SSAValue):
            super().__init__(operands=[arg], result_types=[ComplexType()])

    @irdl_op_definition
    class ImOp(IRDLOperation):
        name = "complex.im"

        arg = operand_def(ComplexType())

        result = result_def(Float64Type())

        def __init__(self, arg: SSAValue):
            super().__init__(operands=[arg], result_types=[ComplexType()])

    @irdl_op_definition
    class AddCOp(IRDLOperation):
        name = "complex.add"

        lhs = operand_def(ComplexType())
        rhs = operand_def(ComplexType())

        result = result_def(ComplexType())

        def __init__(self, lhs: SSAValue, rhs: SSAValue):
            super().__init__(operands=[lhs, rhs], result_types=[ComplexType()])


    @irdl_op_definition
    class MulCOp(IRDLOperation):
        name = "complex.mul"

        lhs = operand_def(ComplexType())
        rhs = operand_def(ComplexType())

        result = result_def(ComplexType())

        def __init__(self, lhs: SSAValue, rhs: SSAValue):
            super().__init__(operands=[lhs, rhs], result_types=[ComplexType()])

    @irdl_op_definition
    class NormOp(IRDLOperation):
        name = "complex.norm"

        arg = operand_def(ComplexType())

        result = result_def(Float64Type())

        def __init__(self, arg: SSAValue):
            super().__init__(operands=[arg], result_types=[Float64Type()])
    return AddCOp, ComplexType, CreateOp, ImOp, MulCOp, NormOp, ReOp


@app.cell
def _(
    Add,
    AddCOp,
    AddfOp,
    AddiOp,
    Block,
    Builder,
    ComplexType,
    ConstantOp,
    CreateOp,
    Expr,
    Float,
    Float64Type,
    FloatAttr,
    ForOp,
    Function,
    I,
    InsertPoint,
    Integer,
    IntegerAttr,
    IntegerType,
    Mul,
    MulCOp,
    MulfOp,
    MuliOp,
    NormOp,
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
    Norm = Function("Norm", real=True)

    def get_type(expr: Expr) -> IntegerType | Float64Type:
        if expr.is_integer:
            return IntegerType(64)
        elif expr.is_real:
            return Float64Type()
        else:
            return ComplexType()

    def emit_op(expr: Expr, builder: Builder, args: dict[str, SSAValue], expected_type: IntegerType | Float64Type | ComplexType):
        # Handle conversions from integer to float
        if isinstance(expected_type, Float64Type) and isinstance(get_type(expr), IntegerType):
            res = emit_op(expr, builder, args, IntegerType(64))
            convert_op = SIToFPOp(res, Float64Type())
            builder.insert(convert_op)
            return convert_op.result

        if isinstance(expected_type, ComplexType) and isinstance(get_type(expr), Float64Type):
            res = emit_op(expr, builder, args, Float64Type())
            zero = ConstantOp(FloatAttr(0, Float64Type()))
            builder.insert(zero)
            create = CreateOp(res, zero.result)
            builder.insert(create)
            return create.result

        if expected_type != get_type(expr):
            raise ValueError(f"Wrong typing, expected {expected_type} but got {get_type(expr)} for expression {expr}")

        if expr == I:
            zero = ConstantOp(FloatAttr(float(0), Float64Type()))
            one = ConstantOp(FloatAttr(float(1), Float64Type()))
            builder.insert(zero)
            builder.insert(one)
            create = CreateOp(zero.result, one.result)
            builder.insert(create)
            return create.result

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
            elif isinstance(expected_type, Float64Type):
                add_op = AddfOp(lhs, rhs)
            else:
                add_op = AddCOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Mul:
            lhs = emit_op(expr.args[0], builder, args, expected_type)
            rhs = emit_op(expr.args[1], builder, args, expected_type)
            if isinstance(expected_type, IntegerType):
                add_op = MuliOp(lhs, rhs)
            elif isinstance(expected_type, Float64Type):
                add_op = MulfOp(lhs, rhs)
            else:
                add_op = MulCOp(lhs, rhs)
            builder.insert(add_op)
            return add_op.result

        if expr.func == Pow:
            assert isinstance(expected_type, Float64Type)
            lhs = emit_op(expr.args[0], builder, args, expected_type)
            rhs = emit_op(expr.args[1], builder, args, expected_type)
            pow_op = PowfOp(lhs, rhs)
            builder.insert(pow_op)
            return pow_op.result

        if expr.func == Norm:
            assert isinstance(expected_type, Float64Type)
            arg = emit_op(expr.args[0], builder, args, ComplexType())
            norm = NormOp(arg)
            builder.insert(norm)
            return norm.result

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
    return Norm, emit_op, get_type


@app.cell
def _(
    Builder,
    Expr,
    FuncOp,
    I,
    InsertPoint,
    ModuleOp,
    Norm,
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
    test(Norm(x + y*I))
    test(Norm(x + y*I) * Norm(z + t*I))
    test(Norm((x + y*I) * (z + t*I)))
    return a, b, emit_ir, t, test, x, y, z


@app.cell
def _(mo):
    mo.md("""# Optimizations""")
    return


@app.cell
def _(
    AddfOp,
    ConstantOp,
    Expr,
    Float64Type,
    FloatAttr,
    GreedyRewritePatternApplier,
    I,
    MulfOp,
    Norm,
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
    test_with_opts(Norm(x + y*I) * Norm(z + t*I))
    test_with_opts(Norm((x + y*I) * (z + t*I)))
    return (
        AddTimesMinusOnePattern,
        Pow2Pattern,
        SIToFPConstantPattern,
        test_with_opts,
    )


if __name__ == "__main__":
    app.run()
