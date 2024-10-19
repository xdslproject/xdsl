from xdsl.context import MLContext
from xdsl.dialects import builtin, eqsat
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddCostEclass(RewritePattern):
    """
    Add cost to each operator in an e-class.
    """

    EQSAT_COST_LABEL = "eqsat_cost"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: eqsat.EClassOp, rewriter: PatternRewriter):
        for operand in op.operands:
            if not isinstance(operand, OpResult):
                # only add costs to operators
                continue

            # Add cost attribute to operators
            operation = operand.op
            operation.attributes[self.EQSAT_COST_LABEL] = IntAttr(1)


class EqsatAddCosts(ModulePass):
    """
    Replace all eqsat.eclass operations in an MLIR program.

    Input example:
       ```mlir
        func.func @test(%a : index, %b : index) -> (index) {
            %a_eq   = eqsat.eclass %a : index
            %one    = arith.constant 1 : index
            %one_eq = eqsat.eclass %one : index
            %amul = arith.muli %a_eq, %one_eq   : index

            %out  = eqsat.eclass %amul, %a_eq : index
            func.return %out : index
        }
       ```
    Output example:
        ```mlir
        builtin.module {
            func.func @test(%a : index, %b : index) -> index {
              %a_eq = eqsat.eclass %a {"eqsat_cost" = #builtin.int<1>} : index
              %one = arith.constant {"eqsat_cost" = #builtin.int<1>} 1 : index
              %one_eq = eqsat.eclass %one : index
              %amul = arith.muli %a_eq, %one_eq : index
              %out = eqsat.eclass %amul, %a_eq : index
              func.return %out : index
            }
        }
        ```
    """

    name = "eqsat-add-costs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AddCostEclass()).rewrite_module(op)
