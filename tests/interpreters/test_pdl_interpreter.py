from io import StringIO

from xdsl.ir import BlockArgument, MLContext, OpResult, Operation
from xdsl.dialects import arith, func, pdl
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr
from xdsl.pattern_rewriter import (PatternRewriter, RewritePattern,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker)
from xdsl.interpreter import Interpreter

from xdsl.interpreters.experimental.pdl import PDLFunctions


class SwapInputs(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
        if not isinstance(op.lhs, OpResult):
            return
        if not isinstance(op.lhs.op, arith.Addi):
            return
        new_op = arith.Addi.get(op.rhs, op.lhs)
        rewriter.replace_op(op, [new_op])


def test_rewrite_python():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()

    PatternRewriteWalker(SwapInputs(),
                         apply_recursively=False).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)


def test_rewrite_pdl():
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

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Constant.from_int_and_width(1, 32)
    b3 = arith.Addi.get(b2.result, b1.result)
    b4 = arith.Addi.get(b3.result, b0.result)
    b5 = func.Return.get(b4.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3, b4, b5])

    return ir_module


def swap_arguments_output():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Constant.from_int_and_width(1, 32)
    b3 = arith.Addi.get(b2.result, b1.result)
    b4 = arith.Addi.get(b0.result, b3.result)
    b5 = func.Return.get(b4.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3, b4, b5])

    return ir_module


def swap_arguments_pdl():
    # The rewrite below matches the second addition as root op

    def pattern(*args: BlockArgument) -> list[Operation]:
        p0 = pdl.OperandOp.get()
        p1 = pdl.OperandOp.get()
        p2 = pdl.TypeOp.get()
        p3 = pdl.OperationOp.get(StringAttr("arith.addi"),
                                 operandValues=[p0.results[0], p1.results[0]],
                                 typeValues=[p2.results[0]])
        p4 = pdl.ResultOp.get(IntegerAttr.from_int_and_width(0, 32),
                              parent=p3.results[0])
        p5 = pdl.OperandOp.get()
        p6 = pdl.OperationOp.get(opName=StringAttr("arith.addi"),
                                 operandValues=[p4.results[0], p5.results[0]],
                                 typeValues=[p2.results[0]])

        def rewrite(*args: BlockArgument) -> list[Operation]:
            p7 = pdl.OperationOp.get(
                opName=StringAttr("arith.addi"),
                operandValues=[p5.results[0], p4.results[0]],
                typeValues=[p2.results[0]])
            o0 = pdl.ReplaceOp.get(p6.results[0], p7.results[0])
            return [p7, o0]

        o1 = pdl.RewriteOp.from_callable(None, p6.results[0], [], rewrite)

        return [p0, p1, p2, p3, p4, p5, p6, o1]

    o2 = pdl.PatternOp.from_callable(IntegerAttr.from_int_and_width(2, 16),
                                     None, pattern)

    pdl_module = ModuleOp.from_region_or_ops([o2])

    return pdl_module


def constant_folding_input():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Addi.get(b0.result, b1.result)
    b3 = func.Return.get(b2.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3])

    return ir_module


def constant_folding_output():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Constant.from_int_and_width(6, 32)
    b3 = func.Return.get(b2.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3])

    return ir_module
