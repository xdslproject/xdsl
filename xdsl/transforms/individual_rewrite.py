from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import HasCanonicalizationPatternsTrait


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
        all_ops = list(op.walk())
        if self.matched_operation_index >= len(all_ops):
            raise ValueError("Matched operation index out of range.")

        matched_operation = all_ops[self.matched_operation_index]
        rewriter = PatternRewriter(matched_operation)

        if matched_operation.name != self.operation_name:
            raise ValueError(
                f"Operation {matched_operation.name} at index "
                f"{self.matched_operation_index} does not match {self.operation_name}"
            )

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

            for pattern_name, pattern in pattern_by_name.items():
                cloned_op = tuple(module_op.clone().walk())[op_idx]
                rewriter = PatternRewriter(cloned_op)
                pattern.match_and_rewrite(cloned_op, rewriter)
                if rewriter.has_done_action:
                    res.append(
                        ApplyIndividualRewritePass(
                            op_idx, cloned_op.name, pattern_name
                        ),
                    )

        return tuple(res)
