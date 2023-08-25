from xdsl.dialects import riscv
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveRedundantMv(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MVOp, rewriter: PatternRewriter) -> None:
        if (
            op.rd.type == op.rs.type
            and isinstance(op.rd.type, riscv.RISCVRegisterType)
            and op.rd.type.is_allocated
        ):
            rewriter.replace_matched_op([], [op.rs])
