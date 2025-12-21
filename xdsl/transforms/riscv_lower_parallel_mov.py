from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp, SSAValue
from xdsl.ir import Attribute, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException


def _replace_result(
    rewriter: PatternRewriter,
    parallel_mv_op: riscv.ParallelMovOp,
    mv_op: Operation,
):
    """Replace one result of a parallel move, determined by mv_op's output type."""
    rewriter.insert_op(mv_op, InsertPoint.before(rewriter.current_operation))
    for x in parallel_mv_op.outputs:
        if x.type == mv_op.result_types[0]:
            rewriter.replace_all_uses_with(x, mv_op.results[0])
            return
    # Otherwise, we have passed an invalid output type
    raise ValueError(
        f"No output with given register type: cannot find {mv_op.results[0]}."
    )


class ParallelMovPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.ParallelMovOp, rewriter: PatternRewriter):
        input_types = cast(Sequence[riscv.RISCVRegisterType], op.inputs.types)
        output_types = cast(Sequence[riscv.RISCVRegisterType], op.outputs.types)

        if not (
            all(i.is_allocated for i in input_types)
            and all(i.is_allocated for i in output_types)
        ):
            raise PassFailedException("All registers must be allocated")

        if not (
            all(isinstance(i, riscv.IntRegisterType) for i in input_types)
            and all(isinstance(i, riscv.IntRegisterType) for i in output_types)
        ):
            raise PassFailedException("Not implemented: non-integer support")

        # make free registers a list so we can add to it later
        free_registers: list[riscv.IntRegisterType] = []
        if op.free_registers is not None:
            free_registers = [
                i for i in op.free_registers if isinstance(i, riscv.IntRegisterType)
            ]

        num_operands = len(op.operands)

        # We have a graph with nodes as registers and directed edges as moves,
        # pointing from source to destination.
        # Every node has at most 1 in edge since we can't write to a register twice.
        # Therefore the graph forms a directed pseudoforest, which is a group of
        # connected components with at most 1 cycle each.

        # If we ignore the cycles, we will have a forest.
        # For each tree, we need to perform each move such that all out edges of a node
        # are before the in edge, so a post-order traversal.
        # We can do this iteratively by storing processed edges for each node.
        # Then we iterate up the tree from every leaf, stopping whenever we encounter
        # a node where all out edges haven't been processed yet.

        # store the back edges of the graph
        dst_to_src: dict[riscv.RegisterType, SSAValue] = {}
        leaves: set[Attribute] = set(op.outputs.types)
        unprocessed_children: Counter[SSAValue] = Counter()

        for idx, src, dst in zip(
            range(num_operands), op.inputs, op.outputs, strict=True
        ):
            # src.type points to something so it can't be a leaf
            leaves.discard(src.type)

            if src.type == dst.type:
                # Trivial case of moving register to itself.
                # We can ignore all instances of this
                rewriter.replace_all_uses_with(dst, src)
            else:
                dst_to_src[dst.type] = src
                unprocessed_children[src] += 1

        for dst in op.outputs.types:
            if dst not in leaves:
                continue
            # Iterate up the tree by traversing back edges.
            while dst in dst_to_src:
                src = dst_to_src[dst]
                _replace_result(rewriter, op, riscv.MVOp(src, rd=dst))
                unprocessed_children[src] -= 1
                # only continue up the tree if all children were processed
                if unprocessed_children[src]:
                    break
                dst = src.type

            # if dst is a register that has no input, we can use it as a free register.
            if dst not in dst_to_src and isinstance(dst, riscv.IntRegisterType):
                free_registers.append(dst)

        # If we have a cycle in the graph, all trees pointing into the cycle cannot
        # enter the cycle because it will have an unprocessed child from its previous
        # node in the cycle.
        # Therefore, all nodes in the cycle will have one unprocessed child
        for node, children in unprocessed_children.items():
            if children != 0:
                # Find a free integer register.
                # We don't have to modify its value since all the cycles
                # can use the same register.
                if not free_registers:
                    raise PassFailedException(
                        "Not implemented: cyclic moves without free int register."
                    )
                temp_reg = free_registers[0]

                # Break the cycle by using free register
                # move node into the free register
                temp_ssa = riscv.MVOp(node, rd=temp_reg)
                rewriter.insert_op(
                    temp_ssa, InsertPoint.before(rewriter.current_operation)
                )
                # we have now created a new chain, with node as the leaf and
                # the temp reg as the root
                unprocessed_children[node] -= 1

                dst = node.type
                # iterate up the chain until we reach the current output
                while True:
                    src = dst_to_src[dst]
                    if src.type == node.type:
                        break
                    _replace_result(rewriter, op, riscv.MVOp(src, rd=dst))
                    unprocessed_children[src] -= 1
                    assert (
                        unprocessed_children[src] == 0
                    )  # nodes should only have 1 child
                    dst = src.type
                # finish the split mov
                _replace_result(rewriter, op, riscv.MVOp(temp_ssa, rd=dst))
        rewriter.erase_op(op)


@dataclass(frozen=True)
class RISCVLowerParallelMovPass(ModulePass):
    """Lowers ParallelMovOp in a module into separate moves."""

    name = "riscv-lower-parallel-mov"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ParallelMovPattern()).rewrite_module(op)
