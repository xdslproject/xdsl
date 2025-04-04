from xdsl.context import Context
from xdsl.dialects import builtin, eqsat, func
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import DiagnosticException


def insert_eclass_op(op: Operation):
    results = op.results

    if len(results) != 1:
        raise DiagnosticException("Ops with non-single results not handled")

    eclass_op = eqsat.EClassOp(results[0])
    insertion_point = InsertPoint.after(op)
    Rewriter.insert_op(eclass_op, insertion_point)
    results[0].replace_by_if(
        eclass_op.results[0], lambda u: not isinstance(u.operation, eqsat.EClassOp)
    )


def create_egraph_op(f: func.FuncOp) -> eqsat.EGraphOp:
    egraph_block = Block()
    egraph_body = Region(egraph_block)
    for op in f.body.block.ops:
        op.detach()
        egraph_block.add_op(op)
        if isinstance(op, func.ReturnOp):
            yieldop = eqsat.YieldOp(*op.arguments)
            Rewriter.replace_op(op, yieldop)
            op = yieldop
        else:
            insert_eclass_op(op)

    for arg in f.body.block.args:
        eclass_op = eqsat.EClassOp(arg)
        insertion_point = InsertPoint.at_start(egraph_block)
        Rewriter.insert_op(eclass_op, insertion_point)
        arg.replace_by_if(
            eclass_op.results[0], lambda u: not isinstance(u.operation, eqsat.EClassOp)
        )
    egraph_op = eqsat.EGraphOp(f.function_type.outputs.data, egraph_body)
    return egraph_op


class InsertEclassOps(RewritePattern):
    """
    Inserts a `eqsat.eclass` after each operation except module op and function op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        egraph_op = create_egraph_op(op)
        op.body.block.add_op(egraph_op)
        op.body.block.add_op(
            func.ReturnOp(
                *egraph_op.results,
            )
        )


class EqsatCreateEclassesPass(ModulePass):
    """
    Create initial eclasses from an MLIR program.

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
            %a_eq = eqsat.eclass %a : index
            %b_eq = eqsat.eclass %b : index
            %c = arith.addi %a_eq, %b_eq : index
            %c_eq = eqsat.eclass %c : index
            func.return %c_eq : index
        }
        ```
    """

    name = "eqsat-create-eclasses"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([InsertEclassOps()]),
            walk_regions_first=True,
            apply_recursively=False,
        ).rewrite_module(op)
