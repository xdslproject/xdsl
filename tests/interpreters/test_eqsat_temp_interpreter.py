# JC: Disabled this to get the minimal example (without eqsat example) working first


# from io import StringIO
#
# from xdsl.builder import Builder, ImplicitBuilder
# from xdsl.context import MLContext
# from xdsl.dialects import arith, func, pdl
# from xdsl.dialects.builtin import (
#     IntegerAttr,
#     ModuleOp,
#     StringAttr,
# )
# from xdsl.interpreters.experimental.pdl import (
#     PDLRewritePattern,
# )
# from xdsl.pattern_rewriter import (
#     PatternRewriter,
#     PatternRewriteWalker,
#     RewritePattern,
#     op_type_rewrite_pattern,
# )
#
#
# class SwapInputs(RewritePattern):
#
#     """
#     This is an example of matching a + b == b + a in eqsat dialect
#     JC: Here we face two design choices:
#     1. traverse each e-class and match the given pattern
#     2. traverse each e-node and match the given pattern
#     here I took the second approach.
#
# Input example:
#     ```mlir
#     func.func @test(%a : index, %b : index) -> (index) {
#         %a_eq = eqsat.eclass %a : index
#         %b_eq = eqsat.eclass %b : index
#         %c = arith.addi %a_eq, %b_eq : index
#         %c_eq = eqsat.eclass %c : index
#         func.return %c_eq : index
#     }
#     ```
#
# Output example:
#     ```mlir
#     func.func @test(%a : index, %b : index) -> (index) {
#         %a_eq = eqsat.eclass %a : index
#         %b_eq = eqsat.eclass %b : index
#         %c = arith.addi %a_eq, %b_eq : index
#         %c_new = arith.addi %b_eq, %a_eq : index
#         %c_eq = eqsat.eclass %c, %c_new : index
#         func.return %c_eq : index
#     }
#     ```
#     """
#
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
#
#         # This is no longer needed because all the operands of e-nodes are
#         # e-classes, i.e. a result of an e-class op.
#         # if not isinstance(op.lhs, OpResult):
#         #     return
#
#         # JC: Not sure why this was needed... deleted.
#         # if not isinstance(op.lhs.op, arith.Addi):
#         #     return
#
#         # Check if the pattern has already been rewritten
#         eclass = op.result.op
#         for arg in eclass.args:
#             # Skip the current op
#             if arg.op == op:
#                 continue
#             if isinstance(arg.op, arith.Addi):
#                 if arg.op.rhs == op.lhs and arg.op.lhs == op.rhs:
#                     return
#
#         # Create a new expression
#         new_op = arith.Addi(op.rhs, op.lhs)
#         new_eclass_op = eqsat.EClass(eclass.args + [new_op])
#         rewriter.replace_op(eclass, [new_eclass_op])
#
#
# def test_rewrite_swap_inputs_python():
#     input_module = swap_arguments_input()
#     output_module = swap_arguments_output()
#
#     PatternRewriteWalker(SwapInputs(), apply_recursively=False).rewrite_module(
#         input_module
#     )
#
#     assert input_module.is_structurally_equivalent(output_module)
#
#
# def test_rewrite_swap_inputs_pdl():
#     input_module = swap_arguments_input()
#     output_module = swap_arguments_output()
#     rewrite_module = swap_arguments_pdl()
#
#     pdl_rewrite_op = next(
#         op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
#     )
#
#     stream = StringIO()
#
#     ctx = MLContext()
#     ctx.load_dialect(arith.Arith)
#
#     PatternRewriteWalker(
#         PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
#         apply_recursively=False,
#     ).rewrite_module(input_module)
#
#     assert input_module.is_structurally_equivalent(output_module)
#
#
# def swap_arguments_input():
#     @ModuleOp
#     @Builder.implicit_region
#     def ir_module():
#         with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
#             x = arith.Constant.from_int_and_width(4, 32).result
#             y = arith.Constant.from_int_and_width(2, 32).result
#             x_y = arith.Addi(x, y).result
#             func.Return(x_y)
#
#     return ir_module
#
#
# def swap_arguments_output():
#     @ModuleOp
#     @Builder.implicit_region
#     def ir_module():
#         with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
#             x = arith.Constant.from_int_and_width(4, 32).result
#             y = arith.Constant.from_int_and_width(2, 32).result
#             y_x = arith.Addi(y, x).result
#             func.Return(y_x)
#
#     return ir_module
#
#
# def swap_arguments_pdl():
#     # The rewrite below matches the second addition as root op
#
#     @ModuleOp
#     @Builder.implicit_region
#     def pdl_module():
#         with ImplicitBuilder(pdl.PatternOp(2, None).body):
#             x = pdl.OperandOp().value
#             y = pdl.OperandOp().value
#             pdl_type = pdl.TypeOp().result
#
#             # JC: Here to check if written pattern has already been added
#
#
#
#             x_y_op = pdl.OperationOp(
#                 StringAttr("arith.addi"), operand_values=[x, y], type_values=[pdl_type]
#             ).op
#             x_y = pdl.ResultOp(IntegerAttr(0, 32), parent=x_y_op).val
#             z = pdl.OperandOp().value
#             x_y_z_op = pdl.OperationOp(
#                 op_name=StringAttr("arith.addi"),
#                 operand_values=[x_y, z],
#                 type_values=[pdl_type],
#             ).op
#
#             # JC: Here instead of replacing the addi op, replace the eclass op
#
#             with ImplicitBuilder(pdl.RewriteOp(x_y_z_op).body):
#                 z_x_y_op = pdl.OperationOp(
#                     StringAttr("arith.addi"),
#                     operand_values=[z, x_y],
#                     type_values=[pdl_type],
#                 ).op
#                 pdl.ReplaceOp(x_y_z_op, z_x_y_op)
#
#     return pdl_module


