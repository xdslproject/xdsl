from xdsl.dialects import pdl
from xdsl.ir import OpResult, SSAValue


class PatternAnalyzer:
    """Analyzes PDL patterns and extracts predicates"""

    def detect_roots(self, pattern: pdl.PatternOp) -> list[OpResult[pdl.OperationType]]:
        """Detect root operations in a pattern"""
        used: set[SSAValue] = set()

        for operation_op in pattern.body.ops:
            if not isinstance(operation_op, pdl.OperationOp):
                continue
            for operand in operation_op.operand_values:
                result_op = operand.owner
                if isinstance(result_op, pdl.ResultOp | pdl.ResultsOp):
                    used.add(result_op.parent_)

        rewriter = pattern.body.block.last_op
        assert isinstance(rewriter, pdl.RewriteOp)
        if rewriter.root is not None:
            if rewriter.root in used:
                used.remove(rewriter.root)

        roots = [
            op.op
            for op in pattern.body.ops
            if isinstance(op, pdl.OperationOp) and op.op not in used
        ]
        return roots
