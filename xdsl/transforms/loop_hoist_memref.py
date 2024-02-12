from dataclasses import dataclass

from xdsl.dialects import builtin, memref, scf
from xdsl.ir import Block, Operation, Region
from xdsl.irdl import Operand
from xdsl.passes import MLContext, ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.utils import (
    find_same_target_store,
    get_operation_at_index,
    is_loop_dependent,
)


@dataclass
class LoopHoistMemref(RewritePattern):
    """Hoist pairs of memref.loads and memref.stores out of a loop."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        for_op: scf.For,
        rewriter: PatternRewriter,
    ) -> None:
        if for_op.parent_block() is None:
            return

        parent_block = for_op.body.block

        loads = [op for op in parent_block.ops if isinstance(op, memref.Load)]

        if not loads:
            return

        # filter out multiple loads from the same location
        load_locs = [load.memref for load in loads]
        dup_load_locs = [loc for loc in set(load_locs) if load_locs.count(loc) > 1]
        loads = [load for load in loads if load.memref not in dup_load_locs]

        load_store_pairs: dict[memref.Load, memref.Store] = {}

        for load in loads:
            if (
                (store := find_same_target_store(load))
                and parent_block.get_operation_index(load)
                < parent_block.get_operation_index(store)
                and not any(is_loop_dependent(idx, for_op) for idx in load.indices)
            ):
                load_store_pairs[load] = store

        # filter out stores using the same value
        store_vals = [store.value for store in load_store_pairs.values()]
        dup_store_vals = [val for val in store_vals if store_vals.count(val) > 1]
        load_store_pairs = {
            load: store
            for load, store in load_store_pairs.items()
            if store.value not in dup_store_vals
        }

        if not load_store_pairs:
            return

        # hoist new loads before the current loop
        new_loads = [load.clone() for load in load_store_pairs.keys()]
        rewriter.insert_op_before(new_loads, for_op)

        ld_indices = [
            parent_block.get_operation_index(load) for load in load_store_pairs.keys()
        ]
        st_indices = [
            parent_block.get_operation_index(store)
            for store in load_store_pairs.values()
        ]

        new_body = Region()
        block_map: dict[Block, Block] = {}
        for_op.body.clone_into(new_body, None, None, block_map)

        new_block_args = [
            new_body.block.insert_arg(new_load.res.type, len(new_body.block.args))
            for new_load in new_loads
        ]

        new_parent_block = block_map[parent_block]

        toerase_ops: list[Operation] = []
        for new_block_arg, idx in zip(new_block_args, ld_indices):
            interim_load = get_operation_at_index(new_parent_block, idx)
            assert isinstance(interim_load, memref.Load)
            interim_load.res.replace_by(new_block_arg)
            toerase_ops.append(interim_load)

        new_yield_vals: list[Operand] = []
        for idx in st_indices:
            interim_store = get_operation_at_index(new_parent_block, idx)
            assert isinstance(interim_store, memref.Store)
            new_yield_vals.append(interim_store.value)
            toerase_ops.append(interim_store)

        for op in toerase_ops:
            op.detach()
            op.erase()

        # yield the value that was used in the old store
        assert new_body.block.last_op is not None
        rewriter.replace_op(new_body.block.last_op, scf.Yield(*new_yield_vals))

        new_for_op = scf.For(for_op.lb, for_op.ub, for_op.step, new_loads, new_body)

        # use yielded results of new loop in stores after the loop
        new_stores = [
            memref.Store.get(new_for_op.res[idx], store.memref, store.indices)
            for idx, store in enumerate(load_store_pairs.values())
        ]
        rewriter.insert_op_after(new_stores, for_op)

        rewriter.insert_op_before(new_for_op, for_op)
        rewriter.erase_op(for_op)


@dataclass
class LoopHoistMemrefPass(ModulePass):
    name = "loop-hoist-memref"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LoopHoistMemref(),
                ]
            ),
            walk_regions_first=True,
            apply_recursively=True,
        ).rewrite_module(op)
