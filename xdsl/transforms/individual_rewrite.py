from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.utils.op_selector import OpSelector


@dataclass(frozen=True)
class ApplyIndividualRewritePass(ModulePass):
    """
    Module pass representing the application of an individual rewrite pattern to a module.

    Matches the operation at the provided index within the module and applies the rewrite
    pattern specified by the operation and pattern names.
    """

    name = "apply-individual-rewrite"

    matched_operation_index: int = field()
    operation_name: str = field()
    pattern_name: str = field()

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        matched_operation = OpSelector(
            self.matched_operation_index, self.operation_name
        ).get_op(op)
        rewriter = PatternRewriter(matched_operation)

        for trait in matched_operation.get_traits_of_type(
            HasCanonicalizationPatternsTrait
        ):
            for pattern in trait.get_canonicalization_patterns():
                if type(pattern).__name__ == self.pattern_name:
                    pattern.match_and_rewrite(matched_operation, rewriter)
                    if not rewriter.has_done_action:
                        raise ValueError(
                            f"Invalid rewrite ({self.pattern_name}) for operation "
                            f"({matched_operation}) at location "
                            f"{self.matched_operation_index}."
                        )
                    return

        raise ValueError(
            f"Pattern name {self.pattern_name} not found for the provided operation name."
        )

    @classmethod
    def schedule_space(cls, ctx: Context, module_op: ModuleOp):
        res: list[ApplyIndividualRewritePass] = []

        for op_idx, matched_op in enumerate(module_op.walk()):
            if (
                trait := matched_op.get_trait(HasCanonicalizationPatternsTrait)
            ) is None:
                continue

            pattern_by_name = {
                type(pattern).__name__: pattern
                for pattern in trait.get_canonicalization_patterns()
            }
            selector = OpSelector(op_idx, matched_op.name)

            for pattern_name, pattern in pattern_by_name.items():
                cloned_op = selector.get_op(module_op.clone())
                rewriter = PatternRewriter(cloned_op)
                pattern.match_and_rewrite(cloned_op, rewriter)
                if rewriter.has_done_action:
                    res.append(
                        ApplyIndividualRewritePass(
                            op_idx, cloned_op.name, pattern_name
                        ),
                    )

        return tuple(res)
