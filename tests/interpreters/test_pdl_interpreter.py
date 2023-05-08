from io import StringIO
from xdsl.builder import Builder

from xdsl.ir import MLContext, OpResult
from xdsl.dialects import arith, func, pdl
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, StringAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
)
from xdsl.interpreter import Interpreter

from xdsl.interpreters.experimental.pdl import PDLFunctions


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

    stream = StringIO()
    interpreter = Interpreter(rewrite_module, file=stream)
    ctx = MLContext()
    ctx.register_dialect(arith.Arith)

    pdl_ft = PDLFunctions(ctx, input_module)
    interpreter.register_implementations(pdl_ft)

    interpreter.run_module()

    assert input_module.is_structurally_equivalent(output_module)


def swap_arguments_input():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        @Builder.implicit_region
        def impl():
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            z = arith.Constant.from_int_and_width(1, 32).result
            x_y = arith.Addi(x, y).result
            x_y_z = arith.Addi(x_y, z).result
            func.Return.get(x_y_z)

        func.FuncOp.from_region("impl", [], [], impl)

    return ir_module


def swap_arguments_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        @Builder.implicit_region
        def impl():
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            z = arith.Constant.from_int_and_width(1, 32).result
            x_y = arith.Addi(x, y).result
            z_x_y = arith.Addi(z, x_y).result
            func.Return.get(z_x_y)

        func.FuncOp.from_region("impl", [], [], impl)

    return ir_module


def swap_arguments_pdl():
    # The rewrite below matches the second addition as root op

    @Builder.implicit_region
    def pattern_region():
        x = pdl.OperandOp().value
        y = pdl.OperandOp().value
        typ = pdl.TypeOp().result

        x_y_op = pdl.OperationOp(
            StringAttr("arith.addi"), operand_values=[x, y], type_values=[typ]
        ).op
        x_y = pdl.ResultOp(IntegerAttr.from_int_and_width(0, 32), parent=x_y_op).val
        z = pdl.OperandOp().value
        x_y_z_op = pdl.OperationOp(
            op_name=StringAttr("arith.addi"), operand_values=[x_y, z], type_values=[typ]
        ).op

        @Builder.implicit_region
        def rewrite_region():
            z_x_y_op = pdl.OperationOp(
                StringAttr("arith.addi"), operand_values=[z, x_y], type_values=[typ]
            ).op
            pdl.ReplaceOp(x_y_z_op, z_x_y_op)

        pdl.RewriteOp(None, x_y_z_op, [], rewrite_region)

    pattern = pdl.PatternOp(IntegerAttr.from_int_and_width(2, 16), None, pattern_region)

    pdl_module = ModuleOp([pattern])

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

    stream = StringIO()
    interpreter = Interpreter(rewrite_module, file=stream)
    ctx = MLContext()
    ctx.register_dialect(arith.Arith)

    pdl_ft = PDLFunctions(ctx, input_module)
    interpreter.register_implementations(pdl_ft)

    interpreter.run_module()

    assert input_module.is_structurally_equivalent(output_module)


def add_zero_input():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        @Builder.implicit_region
        def impl():
            x = arith.Constant.from_int_and_width(4, 32)
            y = arith.Constant.from_int_and_width(0, 32)
            z = arith.Addi(x, y)
            func.Return.get(z)

        func.FuncOp.from_region("impl", [], [], impl)

    return ir_module


def add_zero_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        @Builder.implicit_region
        def impl():
            x = arith.Constant.from_int_and_width(4, 32)
            _y = arith.Constant.from_int_and_width(0, 32)
            func.Return.get(x)

        func.FuncOp.from_region("impl", [], [], impl)

    return ir_module


def add_zero_pdl():
    # The rewrite below matches the second addition as root op

    @Builder.implicit_region
    def pattern_region():
        # Type i32
        pdl_i32 = pdl.TypeOp().result

        # LHS: i32
        lhs = pdl.OperandOp().results[0]

        # Constant 0: i32
        zero = pdl.AttributeOp(value=IntegerAttr.from_int_and_width(0, 32)).results[0]
        rhs_op = pdl.OperationOp(
            op_name=StringAttr("arith.constant"),
            attribute_value_names=ArrayAttr([StringAttr("value")]),
            attribute_values=[zero],
            type_values=[pdl_i32],
        ).op
        rhs = pdl.ResultOp(IntegerAttr.from_int_and_width(0, 32), parent=rhs_op).val

        # LHS + 0
        sum = pdl.OperationOp(
            StringAttr("arith.addi"), operand_values=[lhs, rhs], type_values=[pdl_i32]
        ).op

        @Builder.implicit_region
        def rewrite_region():
            pdl.ReplaceOp(sum, repl_values=[lhs])

        pdl.RewriteOp(None, sum, [], rewrite_region)

    pattern = pdl.PatternOp(IntegerAttr.from_int_and_width(2, 16), None, pattern_region)

    pdl_module = ModuleOp([pattern])

    return pdl_module
