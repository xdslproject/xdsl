from xdsl.context import Context
from xdsl.dialects import builtin, equivalence, func
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
from xdsl.utils.exceptions import DiagnosticException


def insert_eclass_ops(block: Block):
    # Insert equivalence.class for each operation
    for op in block.ops:
        results = op.results

        # Skip special ops such as return ops
        if isinstance(op, func.ReturnOp):
            continue

        if len(results) != 1:
            raise DiagnosticException("Ops with non-single results not handled")

        eclass_op = equivalence.ClassOp(results[0])
        insertion_point = InsertPoint.after(op)
        Rewriter.insert_op(eclass_op, insertion_point)
        results[0].replace_by_if(
            eclass_op.results[0],
            lambda u: not isinstance(u.operation, equivalence.ClassOp),
        )

    # Insert equivalence.class for each arg
    for arg in block.args:
        eclass_op = equivalence.ClassOp(arg)
        insertion_point = InsertPoint.at_start(block)
        Rewriter.insert_op(eclass_op, insertion_point)
        arg.replace_by_if(
            eclass_op.results[0],
            lambda u: not isinstance(u.operation, equivalence.ClassOp),
        )


class InsertEclassOps(RewritePattern):
    """
    Inserts a `equivalence.class` after each operation except module op and function op.
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
            %a_eq = equivalence.class %a : index
            %b_eq = equivalence.class %b : index
            %c = arith.addi %a_eq, %b_eq : index
            %c_eq = equivalence.class %c : index
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
