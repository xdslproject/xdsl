from xdsl.context import MLContext
from xdsl.dialects import builtin, eqsat, func
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter


def insert_eclass_ops(block: Block):
    # Insert eqsat.eclass for each operation
    for op in block.ops:
        results = op.results
        if len(results) != 1:
            continue
            # TODO: ignore operations with no result for now
            # raise DiagnosticException("Ops with non-single results not handled")

        eclass_op = eqsat.EClassOp(results[0])
        insertion_point = InsertPoint.after(op)
        Rewriter.insert_op(eclass_op, insertion_point)
        results[0].replace_by(eclass_op.results[0])
        # Redirect eclassop operand back to the original value
        # TODO: do we need a `replace_by_except` function, e.g.
        # eclass_op.result.replace_by_except(eclass_op.results[0], [eclass_op.result])
        eclass_op.operands[0] = results[0]

    # Insert eqsat.eclass for each arg
    for arg in block.args:
        eclass_op = eqsat.EClassOp(arg)
        insertion_point = InsertPoint.at_start(block)
        Rewriter.insert_op(eclass_op, insertion_point)
        arg.replace_by(eclass_op.results[0])
        # Redirect eclassop operand back to the original value
        # TODO: do we need a `replace_by_except` function, e.g.
        # arg.replace_by_except(eclass_op.results[0], [arg])
        eclass_op.operands[0] = arg


class InsertEclassOps(RewritePattern):
    """
    Inserts a `eqsat.eclass` after each operation except module op and function op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        insert_eclass_ops(op.body.block)

    # ops, indices = insert_affine_map_ops(op.map, op.indices, [])
    # rewriter.insert_op_before_matched_op(ops)

    # # TODO: add nontemporal=false once that's added to memref
    # # https://github.com/xdslproject/xdsl/issues/1482
    # rewriter.replace_matched_op(memref.Store.get(op.value, op.memref, indices))


class EqsatCreateEclasses(ModulePass):
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

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([InsertEclassOps()]),
            apply_recursively=False,
        ).rewrite_module(op)
