from xdsl.context import MLContext
from xdsl.dialects import builtin, eqsat
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddCostEclass(RewritePattern):
    """
    Add cost to each operator in an e-class.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: eqsat.EClassOp, rewriter: PatternRewriter):
        for operand in op.operands:
            if not isinstance(operand, OpResult):
                # only add costs to operators
                continue

            # Add cost attribute to operators
            operation = operand.op
            operation.attributes["cost"] = IntAttr(1)


class EqsatAddCosts(ModulePass):
    """
    Replace all eqsat.eclass operations in an MLIR program.

    Input example:
       ```mlir
       func.func @test(%a : index, %b : index) -> (index) {
            %a_eq = eqsat.eclass %a : index
            %b_eq = eqsat.eclass %b : index
            %c_ab = arith.addi %a_eq, %b_eq   : index
            %c_ba = arith.addi %b_eq, %a_eq   : index
            %c_eq = eqsat.eclass %c_ab, %c_ba : index
            func.return %c_eq : index
        }
       ```
    Output example:
        ```mlir
        func.func @test(%a : index, %b : index) -> (index) {
           %c_ab = arith.addi %a, %b : index
           func.return %c_ab : index
        }
        ```
    """

    name = "eqsat-add-costs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddCostEclass(),
                ]
            ),  # list of rewrite patterns
            apply_recursively=True,  # do we apply rewrites in a while loop
        ).rewrite_module(op)
