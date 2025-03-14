import itertools
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from warnings import warn

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def _try_remove_cast_chain(
    op: builtin.UnrealizedConversionCastOp,
    rewriter: PatternRewriter,
    warn_on_failure: bool,
):
    # casts that either have no uses or have at least
    # one user that isn't an unrealized cast.
    finalized_casts = list[builtin.UnrealizedConversionCastOp]()

    # casts with only other unrealized casts as users.
    pending_casts = deque[builtin.UnrealizedConversionCastOp]()

    # casts that neeed to be visited in the DFS traversal of the use-def chain.
    casts_to_visit = [op]

    def gen_all_uses_cast(
        node: builtin.UnrealizedConversionCastOp,
    ) -> Iterable[Use]:
        for result in node.results:
            yield from result.uses

    # DFS traversal of the use-def chain.
    while casts_to_visit:
        cast = casts_to_visit.pop()

        is_live = False
        has_any_uses = False

        for use in gen_all_uses_cast(cast):
            if isinstance(use.operation, builtin.UnrealizedConversionCastOp):
                # early check, it's definitely not an unifiable cast
                if any(
                    r != i
                    for r, i in itertools.zip_longest(
                        use.operation.inputs, cast.results
                    )
                ):
                    if warn_on_failure:
                        warn(
                            f"Unable to remove cast {cast} because "
                            "it is not unifiable with its uses"
                        )
                    return
                casts_to_visit.append(use.operation)
            else:
                is_live = True
            has_any_uses = True

        # check that either there is a trivial cycle we can remove
        # because types are homogeneous (e.g. {A -> B, B -> A})
        # otherwise it means the cast is not unifiable with its uses
        assert len(cast.results) == len(op.inputs)
        has_trivial_cycle = cast.result_types == op.inputs.types
        if is_live and not has_trivial_cycle:
            if warn_on_failure:
                warn(
                    "Unable to remove cast "
                    f"{cast} because it is not unifiable "
                    "with its uses"
                )
            return

        if not has_any_uses or is_live:
            finalized_casts.append(cast)
        else:
            pending_casts.appendleft(cast)

    for cast in filter(lambda c: c.parent is not None, finalized_casts):
        # replace the uses of this cast by the inputs of the root cast
        rewriter.replace_op(cast, [], op.inputs)

    for cast in filter(lambda c: c.parent is not None, pending_casts):
        # remove other casts in the chain in the right order
        rewriter.erase_op(cast)


@dataclass
class ReconcileUnrealizedCastsPattern(RewritePattern):
    """
    Removes the chains of `builtin.unrealized_conversion_cast` operations
    that are no longer necessary and that start with the matched
    `builtin.unrealized_conversion_cast`.
    """

    warn_on_failure: bool = field(default=False, kw_only=True)

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: builtin.UnrealizedConversionCastOp, rewriter: PatternRewriter
    ):
        _try_remove_cast_chain(op, rewriter, self.warn_on_failure)


def reconcile_unrealized_casts(module: ModuleOp, *, warn_on_failure: bool = True):
    """
    Removes all `builtin.unrealized_conversion_cast` operations
    that are not needed anymore in a module.
    """

    PatternRewriteWalker(
        ReconcileUnrealizedCastsPattern(warn_on_failure=warn_on_failure)
    ).rewrite_module(module)


class ReconcileUnrealizedCastsPass(ModulePass):
    name = "reconcile-unrealized-casts"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        reconcile_unrealized_casts(op)
