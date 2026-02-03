from dataclasses import dataclass

from xdsl.dialects import builtin, memref, scf
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import Operand
from xdsl.passes import Context, ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def find_same_target_store(load: memref.LoadOp):
    """
    Find the corresponding store operation (same memeref target) for a load when there
    is only a single one within a block.
    """

    parent_block = load.parent_block()

    if parent_block is None:
        return None

    found_op = None

    for op in parent_block.ops:
        if (
            isinstance(op, memref.StoreOp)
            and op.memref == load.memref
            and op.indices == load.indices
        ):
            if found_op is None:
                found_op = op
            else:
                return None

    return found_op


def is_loop_dependent(val: SSAValue, loop: scf.ForOp):
    """
    Returns true if the SSA value is dependent by the induction varialbe of the loop.

    This is achieved by traversing the SSA use-def chain of the SSA value; if the
    induction variable contributes to the value, then it depends on it.
    """
    worklist: set[SSAValue] = set()
    visited: set[SSAValue] = set()

    worklist.add(val)

    while worklist:
        val = worklist.pop()
        visited.add(val)

        if val is loop.body.block.args[0]:
            return True

        if isinstance(val.owner, Operation):
            for oprnd in val.owner.operands:
                if oprnd not in visited:
                    worklist.add(oprnd)

    return False


@dataclass
class LoopHoistMemRef(RewritePattern):
    """
    Hoist pairs of memref.loads and memref.stores out of a loop.

    This rewrite hoists pairs of memref.load and memref.store operations outside
    of their enclosing scf.loop. The memref operation pair is considered for this rewrite
    if their memref target location is the same and it is constant w.r.t. the induction
    variable of the containing loop.

    The functionality is intentionally very restricted and does not handle:

    - Multiple loads from the same location
    - Multiple stores of the same value and/or to same location
    - Loads must precede stores (i.e., WAR-only dependence)
    - Does not handle only loads or only stores
    - There is no consideration of aliasing (restricted only to same location memrefs)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        for_op: scf.ForOp,
        rewriter: PatternRewriter,
    ) -> None:
        if for_op.parent_block() is None:
            return

        parent_block = for_op.body.block

        loads = [op for op in parent_block.ops if isinstance(op, memref.LoadOp)]

        if not loads:
            return

        # filter out multiple loads from the same location
        load_locs = [load.memref for load in loads]
        dup_load_locs = [loc for loc in set(load_locs) if load_locs.count(loc) > 1]
        loads = [load for load in loads if load.memref not in dup_load_locs]

        load_store_pairs: dict[memref.LoadOp, memref.StoreOp] = {}

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
        rewriter.insert_op(new_loads, InsertPoint.before(for_op))

        new_body = Region()
        block_map: dict[Block, Block] = {}
        for_op.body.clone_into(new_body, None, None, block_map)

        load_map = {
            load: new_load
            for load, new_load in zip(for_op.body.block.ops, new_body.block.ops)
            if isinstance(load, memref.LoadOp) and isinstance(new_load, memref.LoadOp)
        }
        store_map = {
            store: new_store
            for store, new_store in zip(for_op.body.block.ops, new_body.block.ops)
            if isinstance(store, memref.StoreOp)
            and isinstance(new_store, memref.StoreOp)
        }

        new_block_args = [
            new_body.block.insert_arg(new_load.res.type, len(new_body.block.args))
            for new_load in new_loads
        ]

        toerase_ops: list[Operation] = []
        for new_block_arg, load in zip(new_block_args, load_store_pairs.keys()):
            interim_load = load_map[load]
            interim_load.res.replace_by(new_block_arg)
            toerase_ops.append(interim_load)

        new_yield_vals: list[Operand] = []
        for store in load_store_pairs.values():
            interim_store = store_map[store]
            new_yield_vals.append(interim_store.value)
            toerase_ops.append(interim_store)

        for op in toerase_ops:
            op.detach()
            op.erase()

        # yield the value that was used in the old store
        assert new_body.block.last_op is not None
        rewriter.replace_op(new_body.block.last_op, scf.YieldOp(*new_yield_vals))

        new_for_op = scf.ForOp(for_op.lb, for_op.ub, for_op.step, new_loads, new_body)

        # use yielded results of new loop in stores after the loop
        new_stores = [
            memref.StoreOp.get(new_for_op.res[idx], store.memref, store.indices)
            for idx, store in enumerate(load_store_pairs.values())
        ]
        rewriter.insert_op(new_stores, InsertPoint.after(for_op))

        rewriter.insert_op(new_for_op, InsertPoint.before(for_op))
        rewriter.erase_op(for_op)


@dataclass(frozen=True)
class LoopHoistMemRefPass(ModulePass):
    name = "loop-hoist-memref"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LoopHoistMemRef(),
            walk_regions_first=True,
            apply_recursively=True,
        ).rewrite_module(op)
