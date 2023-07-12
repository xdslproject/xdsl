from io import StringIO

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, func, pdl
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, StringAttr
from xdsl.interpreters.experimental.pdl import PDLRewritePattern
from xdsl.ir import MLContext, OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class SwapInputs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
        if not isinstance(op.lhs, OpResult):
            return
        if not isinstance(op.lhs.op, arith.Addi):
            return
        new_op = arith.Addi(op.rhs, op.lhs)
        rewriter.replace_op(op, [new_op])


def test_rewrite_swap_inputs_python():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()

    PatternRewriteWalker(SwapInputs(), apply_recursively=False).rewrite_module(
        input_module
    )

    assert input_module.is_structurally_equivalent(output_module)


def test_rewrite_swap_inputs_pdl():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()
    rewrite_module = swap_arguments_pdl()

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    stream = StringIO()

    ctx = MLContext()
    ctx.register_dialect(arith.Arith)

    PatternRewriteWalker(
        PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
        apply_recursively=False,
    ).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)


def swap_arguments_input():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            z = arith.Constant.from_int_and_width(1, 32).result
            x_y = arith.Addi(x, y).result
            x_y_z = arith.Addi(x_y, z).result
            func.Return(x_y_z)

    return ir_module


def swap_arguments_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            z = arith.Constant.from_int_and_width(1, 32).result
            x_y = arith.Addi(x, y).result
            z_x_y = arith.Addi(z, x_y).result
            func.Return(z_x_y)

    return ir_module


def swap_arguments_pdl():
    # The rewrite below matches the second addition as root op

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            x = pdl.OperandOp().value
            y = pdl.OperandOp().value
            pdl_type = pdl.TypeOp().result

            x_y_op = pdl.OperationOp(
                StringAttr("arith.addi"), operand_values=[x, y], type_values=[pdl_type]
            ).op
            x_y = pdl.ResultOp(IntegerAttr(0, 32), parent=x_y_op).val
            z = pdl.OperandOp().value
            x_y_z_op = pdl.OperationOp(
                op_name=StringAttr("arith.addi"),
                operand_values=[x_y, z],
                type_values=[pdl_type],
            ).op

            with ImplicitBuilder(pdl.RewriteOp(x_y_z_op).body):
                z_x_y_op = pdl.OperationOp(
                    StringAttr("arith.addi"),
                    operand_values=[z, x_y],
                    type_values=[pdl_type],
                ).op
                pdl.ReplaceOp(x_y_z_op, z_x_y_op)

    return pdl_module


class AddZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
        if not isinstance(op.rhs, OpResult):
            return
        if not isinstance(op.rhs.op, arith.Constant):
            return
        rhs = op.rhs.op
        if not isinstance(rhs.value, IntegerAttr):
            return
        if rhs.value.value.data != 0:
            return
        rewriter.replace_matched_op([], new_results=[op.lhs])


def test_rewrite_add_zero_python():
    input_module = add_zero_input()
    output_module = add_zero_output()

    PatternRewriteWalker(AddZero(), apply_recursively=False).rewrite_module(
        input_module
    )

    assert input_module.is_structurally_equivalent(output_module)


def test_rewrite_add_zero_pdl():
    input_module = add_zero_input()
    output_module = add_zero_output()
    rewrite_module = add_zero_pdl()
    # input_module.verify()
    # output_module.verify()
    rewrite_module.verify()

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    stream = StringIO()

    ctx = MLContext()
    ctx.register_dialect(arith.Arith)

    PatternRewriteWalker(
        PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
        apply_recursively=False,
    ).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)


def add_zero_input():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32)
            y = arith.Constant.from_int_and_width(0, 32)
            z = arith.Addi(x, y)
            func.Return(z)

    return ir_module


def add_zero_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32)
            _y = arith.Constant.from_int_and_width(0, 32)
            func.Return(x)

    return ir_module


def add_zero_pdl():
    # The rewrite below matches the second addition as root op
    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            # Type i32
            pdl_i32 = pdl.TypeOp().result

            # LHS: i32
            lhs = pdl.OperandOp().results[0]

            # Constant 0: i32
            zero = pdl.AttributeOp(value=IntegerAttr(0, 32)).results[0]
            rhs_op = pdl.OperationOp(
                op_name=StringAttr("arith.constant"),
                attribute_value_names=ArrayAttr([StringAttr("value")]),
                attribute_values=[zero],
                type_values=[pdl_i32],
            ).op
            rhs = pdl.ResultOp(IntegerAttr(0, 32), parent=rhs_op).val

            # LHS + 0
            sum = pdl.OperationOp(
                StringAttr("arith.addi"),
                operand_values=[lhs, rhs],
                type_values=[pdl_i32],
            ).op

            with ImplicitBuilder(pdl.RewriteOp(sum).body):
                pdl.ReplaceOp(sum, repl_values=[lhs])

    return pdl_module
