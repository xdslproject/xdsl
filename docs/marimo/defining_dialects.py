# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xdsl==0.27.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
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
    )
    from sympy.core.symbol import Symbol

    from xdsl.ir import (
        Operation,
        SSAValue,
        Region,
        Block,
        ParametrizedAttribute,
        Attribute,
        TypeAttribute,
    )
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
        IRDLOperation,
        irdl_attr_definition,
        operand_def,
        result_def,
        traits_def,
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
    )
    from xdsl.dialects.scf import ForOp, YieldOp
    from xdsl.dialects.math import PowFOp, SqrtOp
    from xdsl.builder import Builder, InsertPoint
    from xdsl.transforms.common_subexpression_elimination import cse
    from xdsl.transforms.dead_code_elimination import dce
    return (
        Abs,
        Add,
        AddfOp,
        AddiOp,
        Attribute,
        Block,
        Builder,
        ConstantOp,
        Expr,
        Float,
        Float64Type,
        FloatAttr,
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
        Region,
        ReturnOp,
        RewritePattern,
        SIToFPOp,
        SSAValue,
        SubfOp,
        Sum,
        Symbol,
        TypeAttribute,
        YieldOp,
        cse,
        dce,
        irdl_attr_definition,
        irdl_op_definition,
        mo,
        operand_def,
        result_def,
        symbols,
        traits_def,
    )


@app.cell(hide_code=True)
def _(Function):
    Norm = Function("Norm", real=True)
    return (Norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Exercise: Add a `complex` dialect

    Your task is to:

    * Write a new set of `complex` operations
    * Add a lowering from `complex` to `arith`
    * Add optimizations for the `complex` dialect
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Defining the `complex` dialect

    You should write the following operations:

    * "complex.create": creates a complex value given its real and imaginary float types
    * "complex.re": returns the real part of a complex number
    * "complex.im": returns the imaginary part of a complex number
    * "complex.add": adds two complex numbers
    * "complex,mul": multiples two copmlex numbers
    * "complex.norm": returns the norm of two complex numbers

    Here is the definition of the `!complex.complex` type, and the definition of `complex.re`. Complete it with the other ops:
    """
    )
    return


@app.cell
def _(
    Float64Type,
    IRDLOperation,
    ParametrizedAttribute,
    Pure,
    SSAValue,
    TypeAttribute,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
):
    @irdl_attr_definition
    class ComplexType(ParametrizedAttribute, TypeAttribute):
        name = "complex.complex"

    @irdl_op_definition
    class ReOp(IRDLOperation):
        name = "complex.re"
        traits = traits_def(Pure())

        arg = operand_def(ComplexType())

        result = result_def(Float64Type())

        def __init__(self, arg: SSAValue):
            super().__init__(operands=[arg], result_types=[Float64Type()])

    @irdl_op_definition
    class ImOp(IRDLOperation):
        name = "complex.im"
        traits = traits_def(Pure())

        # Name the fields arg and result

        def __init__(self, arg: SSAValue):
            raise NotImplementedError("ImOp __init__ is not yet implemented")

    @irdl_op_definition
    class CreateOp(IRDLOperation):
        name = "complex.create"
        traits = traits_def(Pure())

        # Name the fields re, im, and result

        def __init__(self, re: SSAValue, im: SSAValue):
            raise NotImplementedError("CreateOp __init__ is not yet implemented")

    @irdl_op_definition
    class AddcOp(IRDLOperation):
        name = "complex.add"
        traits = traits_def(Pure())

        # Name the fields lhs, rhs, and result

        def __init__(self, lhs: SSAValue, rhs: SSAValue):
            raise NotImplementedError("AddcOp __init__ is not yet implemented")

    @irdl_op_definition
    class MulcOp(IRDLOperation):
        name = "complex.mul"
        traits = traits_def(Pure())

        # Name the fields lhs, rhs, and result

        def __init__(self, lhs: SSAValue, rhs: SSAValue):
            raise NotImplementedError("MulcOp __init__ is not yet implemented")

    @irdl_op_definition
    class NormOp(IRDLOperation):
        name = "complex.norm"
        traits = traits_def(Pure())

        # Name the fields arg and result

        def __init__(self, arg: SSAValue):
            raise NotImplementedError("NormOp __init__ is not yet implemented")
    return AddcOp, ComplexType, CreateOp, ImOp, MulcOp, NormOp, ReOp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Solution

    Hidden below is the definition of all operations
    """
    )
    return


@app.cell(hide_code=True)
def _(
    Float64Type,
    IRDLOperation,
    ParametrizedAttribute,
    Pure,
    SSAValue,
    TypeAttribute,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
):
    def dialect_solution():
        @irdl_attr_definition
        class ComplexType(ParametrizedAttribute, TypeAttribute):
            name = "complex.complex"

        @irdl_op_definition
        class CreateOp(IRDLOperation):
            name = "complex.create"

            traits = traits_def(Pure())

            re = operand_def(Float64Type())
            im = operand_def(Float64Type())

            result = result_def(ComplexType())

            def __init__(self, re: SSAValue, im: SSAValue):
                super().__init__(operands=[re, im], result_types=[ComplexType()])

        @irdl_op_definition
        class ReOp(IRDLOperation):
            name = "complex.re"

            traits = traits_def(Pure())

            arg = operand_def(ComplexType())

            result = result_def(Float64Type())

            def __init__(self, arg: SSAValue):
                super().__init__(operands=[arg], result_types=[Float64Type()])

        @irdl_op_definition
        class ImOp(IRDLOperation):
            name = "complex.im"

            traits = traits_def(Pure())

            arg = operand_def(ComplexType())

            result = result_def(Float64Type())

            def __init__(self, arg: SSAValue):
                super().__init__(operands=[arg], result_types=[Float64Type()])

        @irdl_op_definition
        class AddcOp(IRDLOperation):
            name = "complex.add"

            traits = traits_def(Pure())

            lhs = operand_def(ComplexType())
            rhs = operand_def(ComplexType())

            result = result_def(ComplexType())

            def __init__(self, lhs: SSAValue, rhs: SSAValue):
                super().__init__(operands=[lhs, rhs], result_types=[ComplexType()])

        @irdl_op_definition
        class MulcOp(IRDLOperation):
            name = "complex.mul"

            traits = traits_def(Pure())

            lhs = operand_def(ComplexType())
            rhs = operand_def(ComplexType())

            result = result_def(ComplexType())

            def __init__(self, lhs: SSAValue, rhs: SSAValue):
                super().__init__(operands=[lhs, rhs], result_types=[ComplexType()])

        @irdl_op_definition
        class NormOp(IRDLOperation):
            name = "complex.norm"

            traits = traits_def(Pure())

            arg = operand_def(ComplexType())

            result = result_def(Float64Type())

            def __init__(self, arg: SSAValue):
                super().__init__(operands=[arg], result_types=[Float64Type()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Filling the IR emitter with `complex` operations:

    Now that our new operations are defined, we can add support for them in the IR emitter.
    We completed this for you below.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    Builder,
    Expr,
    FuncOp,
    InsertPoint,
    ModuleOp,
    ReturnOp,
    cse,
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

    def print_ir(expr: Expr):
        try:
            # Print the SymPy expression
            print(expr)

            # Converts the SymPy expression to an MLIR `builtin.module` operation
            op = emit_ir(expr)
            cse(op)

            # Check that the operation verifies, and prints the operation
            op.verify()
            print(op)

            # Print a separator
            print("\n\n")
        except NotImplementedError as e:
            print("Error:", e)
    return emit_ir, print_ir


@app.cell(hide_code=True)
def _(Attribute, ComplexType, Expr, Float64Type, IntegerType):
    def get_mlir_type(expr: Expr) -> Attribute:
        if expr.is_integer:
            return IntegerType(64)
        elif expr.is_real:
            return Float64Type()
        else:
            return ComplexType()
    return (get_mlir_type,)


@app.cell(hide_code=True)
def _(
    Abs,
    Add,
    AddcOp,
    AddfOp,
    AddiOp,
    Block,
    Builder,
    CmpfOp,
    ComplexType,
    ConstantOp,
    CreateOp,
    Expr,
    Float,
    Float64Type,
    FloatAttr,
    ForOp,
    I,
    IfOp,
    InsertPoint,
    Integer,
    IntegerAttr,
    IntegerType,
    Mul,
    MulcOp,
    MulfOp,
    MuliOp,
    Norm,
    NormOp,
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
        elif isinstance(type, ComplexType):
            return emit_complex_op(expr, builder, args)

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

        raise NotImplementedError(f"No IR emitter for integer function {expr.func}")

    def emit_complex_op(
        expr: Expr,
        builder: Builder,
        args: dict[Symbol, SSAValue],
    ):
        if isinstance(get_mlir_type(expr), Float64Type):
            res = emit_real_op(expr, builder, args)
            zero = builder.insert(ConstantOp(FloatAttr(0, Float64Type()))).result
            create = builder.insert(CreateOp(res, zero)).result
            return create

        if expr == I:
            zero = builder.insert(ConstantOp(FloatAttr(float(0), Float64Type()))).result
            one = builder.insert(ConstantOp(FloatAttr(float(1), Float64Type()))).result
            create = builder.insert(CreateOp(zero, one)).result
            return create

        if isinstance(expr, Add):
            lhs = emit_complex_op(expr.args[0], builder, args)
            rhs = emit_complex_op(expr.args[1], builder, args)
            res = builder.insert(AddcOp(lhs, rhs)).result
            return res

        if isinstance(expr, Mul):
            lhs = emit_complex_op(expr.args[0], builder, args)
            rhs = emit_complex_op(expr.args[1], builder, args)
            res = builder.insert(MulcOp(lhs, rhs)).result
            return res

        raise NotImplementedError("Not implemented")

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

        if isinstance(expr, Norm):
            arg = emit_complex_op(expr.args[0], builder, args)
            res = builder.insert(NormOp(arg)).result
            return res

        raise NotImplementedError(f"No IR emitter for float function {expr.func}")
    return (emit_op,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here are a few examples of the code that we can generate:""")
    return


@app.cell
def _(symbols):
    x, y, z, t = symbols("x y z t", real=True)
    return t, x, y, z


@app.cell
def _(I, print_ir, x, y):
    print_ir(x + I * y)
    return


@app.cell
def _(I, print_ir, t, x, y, z):
    print_ir((x + I * y) * (z + I * t))
    return


@app.cell
def _(I, Norm, print_ir, x, y):
    print_ir(Norm(x + I * y))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Lowering the complex dialect

    Now that we emitted IR using the `complex` dialect, we can lower it to `arith`.

    To do this, follow these steps:

    * Write the patterns `re(create(x, y)) -> x` and `im(create(x, y))`
    * Write patterns to rewrite `complex.add`, `complex.mul`, and `complex.norm` into `arith` and `complex.create`, `complex.re`, and `complex.im` ops. For instance, `add(x, y)` should be rewritten to `create(re(x) + re(y), im(x) + im(y))`.

    This will effectively lower the `complex` dialect, as all these patterns will be applied until convergence. We give you the pattern `re(create(x, y)) -> x` and the lowering of `complex.mul`
    """
    )
    return


@app.cell
def _(
    AddfOp,
    CreateOp,
    GreedyRewritePatternApplier,
    ImOp,
    MulcOp,
    MulfOp,
    Operation,
    PatternRewriteWalker,
    PatternRewriter,
    ReOp,
    RewritePattern,
    SubfOp,
    cse,
    dce,
):
    # Lower the complex dialect.
    def lower_complex(op: Operation):
        # Hint: Add rewrite patterns in this list
        rewrites = [
            FoldReCreateOp(),
            FoldImCreateOp(),
            LowerAddOp(),
            LowerMulOp(),
            LowerNormOp(),
        ]
        PatternRewriteWalker(GreedyRewritePatternApplier(rewrites)).rewrite_module(op)

        # Run dce and cse after the rewritting
        cse(op)
        dce(op)

    class FoldReCreateOp(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, ReOp):
                return

            if not isinstance(create := op.arg.owner, CreateOp):
                return

            rewriter.replace_op(op, [], new_results=[create.re])

    class FoldImCreateOp(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            # Implement im(create(x, y)) -> y
            return

    class LowerAddOp(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            # Implement the lowering of add
            # The formula is `add(x, y) = (re(x) + re(y)) + i * (im(x) + im(y))
            return

    class LowerMulOp(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, MulcOp):
                return

            re_lhs = rewriter.insert(ReOp(op.lhs)).result
            re_rhs = rewriter.insert(ReOp(op.rhs)).result
            im_lhs = rewriter.insert(ImOp(op.lhs)).result
            im_rhs = rewriter.insert(ImOp(op.rhs)).result

            tmp1 = rewriter.insert(MulfOp(re_lhs, re_rhs)).result
            tmp2 = rewriter.insert(MulfOp(im_lhs, im_rhs)).result
            new_re = rewriter.insert(SubfOp(tmp1, tmp2)).result

            tmp3 = rewriter.insert(MulfOp(re_lhs, im_rhs)).result
            tmp4 = rewriter.insert(MulfOp(im_lhs, re_rhs)).result
            new_im = rewriter.insert(AddfOp(tmp3, tmp4)).result

            create = rewriter.insert(CreateOp(new_re, new_im)).result

            rewriter.replace_op(op, [], new_results=[create])

    class LowerNormOp(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            # Implement the lowering of norm
            # The formula is `norm(z) = (re(z) * re(z) + im(z) * im(z)) ^ 0.5`
            return
    return (lower_complex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can test your code using the `print_ir_with_complex_lowering` function:""")
    return


@app.cell(hide_code=True)
def _(Expr, cse, emit_ir, lower_complex):
    def print_ir_with_complex_lowering(expr: Expr):
        try:
            # Print the SymPy expression
            print(expr)

            # Converts the SymPy expression to an MLIR `builtin.module` operation
            op = emit_ir(expr)
            cse(op)

            # Check that the operation verifies, and prints the operation
            op.verify()
            print("Op before lowering:")
            print(op)

            print("Op after lowering:")
            lower_complex(op)
            op.verify()
            print(op)

            # Print a separator
            print("\n\n")
        except NotImplementedError as e:
            print("Error:", e)
    return (print_ir_with_complex_lowering,)


@app.cell
def _(I, Norm, print_ir_with_complex_lowering, x, y):
    print_ir_with_complex_lowering(Norm(x + I * y))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Solution

    Hidden below is a possible lowering:
    """
    )
    return


@app.cell(hide_code=True)
def _(
    AddcOp,
    AddfOp,
    ConstantOp,
    CreateOp,
    Float64Type,
    FloatAttr,
    GreedyRewritePatternApplier,
    ImOp,
    MulcOp,
    MulfOp,
    NormOp,
    Operation,
    PatternRewriteWalker,
    PatternRewriter,
    PowFOp,
    ReOp,
    RewritePattern,
    SubfOp,
    cse,
    dce,
):
    def solution():
        # Lower the complex dialect.
        def lower_complex(op: Operation):
            # Hint: Add rewrite patterns in this list
            rewrites = [
                FoldReCreateOp(),
                FoldImCreateOp(),
                LowerAddOp(),
                LowerMulOp(),
                LowerNormOp(),
            ]
            PatternRewriteWalker(GreedyRewritePatternApplier(rewrites)).rewrite_module(
                op
            )

            # Run dce and cse after the rewritting
            cse(op)
            dce(op)

        class FoldReCreateOp(RewritePattern):
            def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
                if not isinstance(op, ReOp):
                    return

                if not isinstance(create := op.arg.owner, CreateOp):
                    return

                rewriter.replace_op(op, [], new_results=[create.re])

        class FoldImCreateOp(RewritePattern):
            def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
                if not isinstance(op, ImOp):
                    return

                if not isinstance(create := op.arg.owner, CreateOp):
                    return

                rewriter.replace_op(op, [], new_results=[create.im])

        class LowerAddOp(RewritePattern):
            def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
                if not isinstance(op, AddcOp):
                    return

                re_lhs = rewriter.insert(ReOp(op.lhs)).result
                re_rhs = rewriter.insert(ReOp(op.rhs)).result
                im_lhs = rewriter.insert(ImOp(op.lhs)).result
                im_rhs = rewriter.insert(ImOp(op.rhs)).result
                new_re = rewriter.insert(AddfOp(re_lhs, re_rhs)).result
                new_im = rewriter.insert(AddfOp(im_lhs, im_rhs)).result
                create = rewriter.insert(CreateOp(new_re, new_im)).result

                rewriter.replace_op(op, [], new_results=[create])

        class LowerMulOp(RewritePattern):
            def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
                if not isinstance(op, MulcOp):
                    return

                re_lhs = rewriter.insert(ReOp(op.lhs)).result
                re_rhs = rewriter.insert(ReOp(op.rhs)).result
                im_lhs = rewriter.insert(ImOp(op.lhs)).result
                im_rhs = rewriter.insert(ImOp(op.rhs)).result

                tmp1 = rewriter.insert(MulfOp(re_lhs, re_rhs)).result
                tmp2 = rewriter.insert(MulfOp(im_lhs, im_rhs)).result
                new_re = rewriter.insert(SubfOp(tmp1, tmp2)).result

                tmp3 = rewriter.insert(MulfOp(re_lhs, im_rhs)).result
                tmp4 = rewriter.insert(MulfOp(im_lhs, re_rhs)).result
                new_im = rewriter.insert(AddfOp(tmp3, tmp4)).result

                create = rewriter.insert(CreateOp(new_re, new_im)).result

                rewriter.replace_op(op, [], new_results=[create])

        class LowerNormOp(RewritePattern):
            def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
                if not isinstance(op, NormOp):
                    return

                re = rewriter.insert(ReOp(op.arg)).result
                im = rewriter.insert(ImOp(op.arg)).result

                re_2 = rewriter.insert(MulfOp(re, re)).result
                im_2 = rewriter.insert(MulfOp(im, im)).result

                add = rewriter.insert(AddfOp(re_2, im_2)).result

                half = rewriter.insert(ConstantOp(FloatAttr(0.5, Float64Type()))).result
                pow = rewriter.insert(PowFOp(add, half)).result

                rewriter.replace_op(op, [], new_results=[pow])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Optimizing our dialects

    As a last task, you have to write your own optimizations patterns to improve as much as possible the performance of the generated code. This includes reducing the amount of floating point power and multiplication functions.

    Use `print_ir_with_pipeline` to test your code, it will run an optimization pass, then the lowering of `complex`, then another optimization pass.

    Your objective is to optimize the function `Norm(z1) * Norm(z2)`.

    As a hint, this expression is equivalent to `Norm(z1 * z2)`, which is much easier to express as a `complex` dialect optimization than an `arith` optimization. You may also add other `arith` and `complex` optimizations.
    """
    )
    return


@app.cell(hide_code=True)
def _(Expr, cse, dce, emit_ir, lower_complex, optimize):
    def print_ir_with_pipeline(expr: Expr):
        try:
            # Print the SymPy expression
            print(expr)

            # Converts the SymPy expression to an MLIR `builtin.module` operation
            print("Emitted IR:")
            op = emit_ir(expr)
            cse(op)
            dce(op)
            op.verify()
            print(op)
            print("\n\n")

            print("After first optimization:")
            optimize(op)
            op.verify()
            print(op)
            print("\n\n")

            print("After complex lowering:")
            lower_complex(op)
            op.verify()
            print(op)
            print("\n\n")

            print("After second optimization:")
            optimize(op)
            op.verify()
            print(op)
            print("\n\n")

            # Print a separator
            print("\n\n")
        except NotImplementedError as e:
            print("Error:", e)
    return (print_ir_with_pipeline,)


@app.cell
def _(GreedyRewritePatternApplier, Operation, PatternRewriteWalker, cse, dce):
    # Optimize the `complex` and `arith` dialects
    def optimize(op: Operation):
        # Hint: Add rewrite patterns in this list
        rewrites = []
        PatternRewriteWalker(GreedyRewritePatternApplier(rewrites)).rewrite_module(op)

        # Run cse and dce
        cse(op)
        dce(op)

    # Define new patterns here
    return (optimize,)


@app.cell
def _(I, Norm, print_ir_with_pipeline, t, x, y, z):
    print_ir_with_pipeline(Norm(x + I * y) * Norm(z + I * t))
    return


if __name__ == "__main__":
    app.run()
