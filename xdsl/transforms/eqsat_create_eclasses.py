from collections.abc import Sequence
from itertools import chain

from ordered_set import OrderedSet

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat, func
from xdsl.ir import Block, Region, SSAValue
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


def create_egraph_op(
    f: func.FuncOp, entries: Sequence[SSAValue], exits: Sequence[SSAValue]
) -> eqsat.EGraphOp:
    exits_set = set(exits)
    egraph_block = Block()
    egraph_body = Region(egraph_block)

    for val in chain(entries, exits):
        assert val.owner.parent_op() == f

    for entry in entries:
        eclass_op = eqsat.EClassOp(entry)
        egraph_block.add_op(eclass_op)
        entry.replace_by_if(eclass_op.results[0], lambda u: u.operation != eclass_op)

    egraph_results: OrderedSet[SSAValue] = OrderedSet([])

    for op in f.body.block.ops:
        if op.regions:
            raise DiagnosticException("Ops with regions not handled")
        if all(
            not isinstance(arg.owner, eqsat.EClassOp) or (arg in egraph_results)
            for arg in op.operands
        ):
            continue

        # move op to egraph body:
        op.detach()
        egraph_block.add_op(op)

        if len(op.results) > 1:
            raise DiagnosticException("Ops with multiple results not handled")
        for res in op.results:
            eclassop = eqsat.EClassOp(res)

            if res in exits_set:
                egraph_results.append(eclassop.results[0])

            insertion_point = InsertPoint.after(op)
            Rewriter.insert_op(eclassop, insertion_point)
            res.replace_by_if(eclassop.results[0], lambda u: u.operation != eclassop)

    egraph_block.add_op(
        yield_op := eqsat.YieldOp(
            *egraph_results,
        )
    )
    egraph_op = eqsat.EGraphOp(yield_op.values.types, egraph_body)
    for i, res in enumerate(egraph_results):
        res.replace_by_if(
            egraph_op.results[i], lambda u: u.operation.parent_op() != egraph_op
        )

    return egraph_op


def insert_eclass_ops(block: Block):
    # Insert eqsat.eclass for each operation
    for op in block.ops:
        results = op.results

        # Skip special ops such as return ops
        if isinstance(op, func.ReturnOp):
            continue

        if len(results) != 1:
            raise DiagnosticException("Ops with non-single results not handled")

        eclass_op = eqsat.EClassOp(results[0])
        insertion_point = InsertPoint.after(op)
        Rewriter.insert_op(eclass_op, insertion_point)
        results[0].replace_by_if(
            eclass_op.results[0], lambda u: not isinstance(u.operation, eqsat.EClassOp)
        )

    # Insert eqsat.eclass for each arg
    for arg in block.args:
        eclass_op = eqsat.EClassOp(arg)
        insertion_point = InsertPoint.at_start(block)
        Rewriter.insert_op(eclass_op, insertion_point)
        arg.replace_by_if(
            eclass_op.results[0], lambda u: not isinstance(u.operation, eqsat.EClassOp)
        )


class InsertEclassOps(RewritePattern):
    """
    Inserts a `eqsat.eclass` after each operation except module op and function op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        insert_eclass_ops(op.body.block)


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
            apply_recursively=False,
        ).rewrite_module(op)
