from xdsl.context import MLContext
from xdsl.dialects import builtin, eqsat
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.transforms.eqsat_add_costs import AddCostEclass


class ExtractEclass(RewritePattern):
    """
    Replace each `eqsat.eclass` operation by one of its children.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: eqsat.EClassOp, rewriter: PatternRewriter):
        # Extraction - replace by the operand by the lowest cost
        index = -1
        min_cost = 0
        for ind, operand in enumerate(op.operands):
            if not isinstance(operand, OpResult):
                index = ind
                break
            elif "cost" in operand.op.attributes:
                if operand.op.attributes["cost"].data < min_cost or index == -1:
                    min_cost = operand.op.attributes["cost"].data
                    index = ind
            else:
                # If no cost has been assigned to an operator - assume ???
                index = ind
                break

        assert index != -1

        # Replace the e-class operator by the operand with the minimal cost
        rewriter.replace_op(op, (), (op.operands[index],))


class EqsatExtractExpressions(ModulePass):
    """
    Replace all eqsat.eclass operations in an MLIR program.

    Input example:
       ```mlir
        func.func @test(%a : index, %b : index) -> (index) {
            %a_eq = eqsat.eclass %a : index
            %one  = arith.constant 1 : index
            %amul = arith.muli %a_eq, %one   : index

            %out  = eqsat.eclass %amul, %a_eq : index
            func.return %out : index
        }
       ```
    Output example:
        ```mlir
        func.func @test(%a : index, %b : index) -> index {
            func.return %a : index
        }
        ```
    """

    name = "eqsat-extract-expressions"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [AddCostEclass(), ExtractEclass(), RemoveUnusedOperations()]
            ),  # list of rewrite patterns
            apply_recursively=True,  # do we apply rewrites in a while loop
        ).rewrite_module(op)
