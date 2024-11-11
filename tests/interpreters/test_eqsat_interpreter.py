from io import StringIO

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, eqsat, func, pdl
from xdsl.dialects.builtin import (
    ModuleOp,
    StringAttr,
)
from xdsl.interpreters.eqsat_pdl import EqsatPDLRewritePattern
from xdsl.interpreters.pdl import PDLRewritePattern
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.eqsat_create_eclasses import insert_eclass_ops

"""
    ======================================================
      Working in progress: swapping inputs without eqsat
    ======================================================

     This is an example of matching a + b == b + a in eqsat dialect
     JC: Here we face two design choices:
     1. traverse each e-class and match the given pattern
     2. traverse each e-node and match the given pattern
     here I took the second approach.

     Input example:
        ```mlir
        func.func @test(%a : index, %b : index) -> (index) {
            %c = arith.addi %a, %b : index
            func.return %c : index
        }
        ```

     Output example:
        ```mlir
        func.func @test(%a : index, %b : index) -> (index) {
            %c_new = arith.addi %b, %a : index
            func.return %c_new : index
        }
        ```
"""


class SwapInputs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
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
    ctx.load_dialect(arith.Arith)

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
            x_y = arith.Addi(x, y).result
            func.Return(x_y)

    return ir_module


def swap_arguments_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            y_x = arith.Addi(y, x).result
            func.Return(y_x)

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

            with ImplicitBuilder(pdl.RewriteOp(x_y_op).body):
                y_x_op = pdl.OperationOp(
                    StringAttr("arith.addi"),
                    operand_values=[y, x],
                    type_values=[pdl_type],
                ).op
                pdl.ReplaceOp(x_y_op, y_x_op)

    return pdl_module


def swap_arguments_eqsat_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32).result
            x_ec = eqsat.EClassOp(x)
            y = arith.Constant.from_int_and_width(2, 32).result
            y_ec = eqsat.EClassOp(y)
            x_y = arith.Addi(y_ec, x_ec).result
            y_x = arith.Addi(x_ec, y_ec).result
            r_ec = eqsat.EClassOp(y_x, x_y)
            func.Return(r_ec)

    return ir_module


def test_hello():
    input_module = swap_arguments_input()
    output_module = swap_arguments_eqsat_output()
    rewrite_module = swap_arguments_pdl()

    insert_eclass_ops(next(iter(input_module.body.block.ops)).regions[0].block)

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    ctx = MLContext()
    ctx.load_dialect(arith.Arith)

    PatternRewriteWalker(
        EqsatPDLRewritePattern(pdl_rewrite_op, ctx),
        apply_recursively=False,
    ).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)
