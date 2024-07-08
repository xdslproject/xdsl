from xdsl.dialects import eqsat, func
from xdsl.ir import Block
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import DiagnosticException


def insert_eclass_ops(block: Block):
    for op in block.ops:
        results = op.results
        if len(results) != 1:
            raise DiagnosticException("Ops with non-single results not handled")

        eclass_op = eqsat.EClassOp(results[0])
        Rewriter.insert_op(op, InsertPoint.after(eclass_op))
        eclass_op.results[0].replace_by(eclass_op.result)


class LowerAffineStore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter): ...

    # ops, indices = insert_affine_map_ops(op.map, op.indices, [])
    # rewriter.insert_op_before_matched_op(ops)

    # # TODO: add nontemporal=false once that's added to memref
    # # https://github.com/xdslproject/xdsl/issues/1482
    # rewriter.replace_matched_op(memref.Store.get(op.value, op.memref, indices))
