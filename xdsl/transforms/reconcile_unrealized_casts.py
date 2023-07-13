import itertools
from collections import deque
from typing import Iterable
from warnings import warn

from xdsl.dialects import builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.ir.core import SSAValue, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter

# def _try_remove_cast_chain(op: builtin.UnrealizedConversionCastOp):
#     # casts that either have no uses or have at least
#     # one user that isn't an unrealized cast.
#     finalized_casts = list[builtin.UnrealizedConversionCastOp]()

#     # casts with only other unrealized casts as users.
#     pending_casts = deque[builtin.UnrealizedConversionCastOp]()

#     # casts that neeed to be visited in the DFS traversal of the use-def chain.
#     casts_to_visit = [op]

#     def gen_all_uses_cast(
#         node: builtin.UnrealizedConversionCastOp,
#     ) -> Iterable[Use]:
#         for result in node.results:
#             yield from result.uses

#     # DFS traversal of the use-def chain.
#     while casts_to_visit:
#         cast = casts_to_visit.pop()

#         is_live = False
#         has_any_uses = False

#         for use in gen_all_uses_cast(cast):
#             if isinstance(use.operation, builtin.UnrealizedConversionCastOp):
#                 # early check, it's definitely not an unifiable cast
#                 if any(
#                     r != i
#                     for r, i in itertools.zip_longest(
#                         use.operation.inputs, cast.results
#                     )
#                 ):
#                     warn(
#                         f"Unable to remove cast {cast} because "
#                         "it is not unifiable with its uses"
#                     )
#                     return
#                 casts_to_visit.append(use.operation)
#             else:
#                 is_live = True
#             has_any_uses = True

#         # check that either there is a trivial cycle we can remove
#         # because types are homogeneous (e.g. {A -> B, B -> A})
#         # otherwise it means the cast is not unifiable with its uses
#         assert len(cast.results) == len(op.inputs)
#         has_trivial_cycle = all(
#             r.type == i.type for r, i in zip(cast.results, op.inputs)
#         )
#         if is_live and not has_trivial_cycle:
#             warn(
#                 "Unable to remove cast "
#                 f"{cast} because it is not unifiable "
#                 "with its uses"
#             )
#             return

#         if not has_any_uses or is_live:
#             finalized_casts.append(cast)
#         else:
#             pending_casts.appendleft(cast)

#     for cast in filter(lambda c: c.parent is not None, finalized_casts):
#         # replace the uses of this cast by the inputs of the root cast
#         PatternRewriter(cast).replace_matched_op([], op.inputs)

#     for cast in filter(lambda c: c.parent is not None, pending_casts):
#         # remove other casts in the chain in the right order
#         PatternRewriter(cast).erase_matched_op()


def reconcile_unrealized_casts(module: ModuleOp):
    """
    Removes all `builtin.unrealized_conversion_cast` operations
    that are not needed anymore in a module.
    """
    casts = [
        op for op in module.walk() if isinstance(op, builtin.UnrealizedConversionCastOp)
    ]

    original_by_cast_result = dict[SSAValue, SSAValue]()

    for cast in casts:
        for operand, result in zip(cast.operands, cast.results):
            original = original_by_cast_result.get(operand, operand)
            original_by_cast_result[result] = original

    for result, original in original_by_cast_result.items():
        if result.type == original.type:
            result.replace_by(original)

    # Perform dce for casts in reverse
    for cast in reversed(casts):
        indices_to_keep = [i for i, result in enumerate(cast.results) if result.uses]

        if not indices_to_keep:
            # erase cast
            PatternRewriter(cast).erase_matched_op()
        else:
            warn(
                "Unable to remove cast "
                f"{cast} because it is not unifiable "
                "with its uses"
            )


class ReconcileUnrealizedCastsPass(ModulePass):
    name = "reconcile-unrealized-casts"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        reconcile_unrealized_casts(op)
