from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.traits import HasCanonicalisationPatternsTrait

REWRITE_BY_NAMES: dict[str, dict[str, RewritePattern]] = {
    op.name: {
        pattern.__class__.__name__: pattern
        for pattern in trait.get_canonicalization_patterns()
    }
    for dialect in get_all_dialects().values()
    for op in dialect().operations
    if (trait := op.get_trait(HasCanonicalisationPatternsTrait)) is not None
}


@dataclass
class IndividualRewrite(ModulePass):
    name = "individual-rewrite"

    matched_operation_index: int | None = None
    operation_name: str | None = None
    pattern_name: str | None = None

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        assert self.matched_operation_index is not None
        assert self.operation_name is not None
        assert self.pattern_name is not None

        matched_operation = list(op.walk())[self.matched_operation_index]
        rewriter = PatternRewriter(matched_operation)
        pattern = REWRITE_BY_NAMES[self.operation_name][self.pattern_name]
        pattern.match_and_rewrite(matched_operation, rewriter)
