from dataclasses import dataclass

from xdsl.dialects import builtin, memref, scf
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import Operand
from xdsl.passes import MLContext, ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def _find_corresponding_store(load: memref.Load):
    parent_block = load.parent_block()

    if parent_block is None:
        return None

    found_op = None

    for op in parent_block.ops:
        if (
            isinstance(op, memref.Store)
            and op.memref == load.memref
            and op.indices == load.indices
        ):
            if found_op is None:
                found_op = op
            else:
                return None

    return found_op


def _is_loop_dependent(val: SSAValue, loop: scf.For):
    worklist: set[SSAValue] = set()
    visited: set[SSAValue] = set()

    worklist.add(val)

    while worklist:
        val = worklist.pop()
        if val in visited:
            continue

        visited.add(val)

        if val is loop.body.block.args[0]:
            return True

        if isinstance(val.owner, Operation):
            for oprnd in val.owner.operands:
                if oprnd not in visited:
                    worklist.add(oprnd)
        else:
            for arg in val.owner.args:
                if arg not in visited:
                    worklist.add(arg)

    return False


@dataclass
class LoopHoistMemref(RewritePattern):
    """Hoist memrefs."""

    loop_depth: int | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        for_op: scf.For,
        rewriter: PatternRewriter,
    ) -> None:
        if for_op.parent_block() is None:
            return

        load_ops: list[memref.Load] = []
        for op in for_op.body.ops:
            if isinstance(op, memref.Load):
                load_ops.append(op)

        # not handling multiple loads from the same location FIX
        if len(load_ops) != len(set(load_ops)):
            return

        load_store_pairs: dict[memref.Load, memref.Store] = {}

        for load_op in load_ops:
            if (store_op := _find_corresponding_store(load_op)) and not any(
                _is_loop_dependent(idx, for_op) for idx in load_op.indices
            ):
                load_store_pairs[load_op] = store_op

        if len(load_store_pairs.items()) == 0:
            return

        parent_block = next(iter(load_store_pairs.values())).parent_block()
        if parent_block is None:
            return

        # hoist new loads before the current loop
        new_load_ops = [load_op.clone() for load_op in load_store_pairs.keys()]
        rewriter.insert_op_before(new_load_ops, for_op)

        ld_indices = [
            parent_block.get_operation_index(load_op)
            for load_op in load_store_pairs.keys()
        ]
        st_indices = [
            parent_block.get_operation_index(store_op)
            for store_op in load_store_pairs.values()
        ]

        new_body = Region()
        block_map: dict[Block, Block] = {}
        for_op.body.clone_into(new_body, None, None, block_map)

        new_block_args = [
            new_body.block.insert_arg(new_load_op.res.type, len(new_body.block.args))
            for new_load_op in new_load_ops
        ]

        new_parent_block = block_map[parent_block]

        for new_block_arg, idx in zip(new_block_args, ld_indices):
            interim_load_op = new_parent_block.get_operation_at_index(idx)
            assert isinstance(interim_load_op, memref.Load)
            interim_load_op.res.replace_by(new_block_arg)
            interim_load_op.detach()
            interim_load_op.erase()

        new_yield_vals: list[Operand] = []
        for idx in st_indices:
            idx = idx - 1
            interim_store_op = new_parent_block.get_operation_at_index(idx)
            assert isinstance(interim_store_op, memref.Store)
            new_yield_vals.append(interim_store_op.value)
            interim_store_op.detach()
            interim_store_op.erase()

        # yield the value that was used in the old store
        assert new_body.block.last_op is not None
        rewriter.replace_op(new_body.block.last_op, scf.Yield(*new_yield_vals))

        new_for_op = scf.For(for_op.lb, for_op.ub, for_op.step, new_load_ops, new_body)

        # use yielded results of new loop in stores after the loop
        new_store_ops = [
            memref.Store.get(new_for_op.res[idx], store_op.memref, store_op.indices)
            for idx, store_op in enumerate(load_store_pairs.values())
        ]
        rewriter.insert_op_after(new_store_ops, for_op)

        rewriter.insert_op_before(new_for_op, for_op)
        rewriter.erase_op(for_op)


@dataclass
class HoistMemrefPass(ModulePass):
    name = "memref-hoist"

    loop_depth: int | None = None

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        if self.loop_depth is None or self.loop_depth < 0:
            self.loop_depth = 0

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LoopHoistMemref(),
                ]
            ),
            walk_regions_first=True,
            apply_recursively=True,
        ).rewrite_module(op)
