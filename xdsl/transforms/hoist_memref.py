from dataclasses import dataclass

from xdsl.dialects import memref, builtin, scf
from xdsl.ir import Block, Operation, Region, SSAValue
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


def _get_loop_nest(op: Operation):
    loop_nest: list[scf.For] = []

    while (parent_op := op.parent_op()) is not None:
        if isinstance(parent_op, scf.For):
            loop_nest.append(parent_op)
        op = parent_op

    return tuple(loop_nest)


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
        load_op: memref.Load,
        rewriter: PatternRewriter,
    ) -> None:
        parent_block = load_op.parent_block()

        if parent_block is None:
            return

        loop_nest = _get_loop_nest(load_op)

        if not loop_nest:
            return

        store_op = _find_corresponding_store(load_op)

        if store_op is None:
            return

        # assert parent_block is st_block

        outer_loop = None
        for loop in loop_nest:
            if not any(_is_loop_dependent(idx, loop) for idx in load_op.indices):
                outer_loop = loop

        if outer_loop is None or outer_loop.parent_block() is None:
            return

        # hoist new load before the current loop
        new_ld = load_op.clone()
        rewriter.insert_op_before(new_ld, outer_loop)

        ld_idx = parent_block.get_operation_index(load_op)
        st_idx = parent_block.get_operation_index(store_op)

        new_body = Region()
        block_map: dict[Block, Block] = {}
        outer_loop.body.clone_into(new_body, None, None, block_map)

        new_block_arg = new_body.block.insert_arg(
            new_ld.res.type, len(new_body.block.args)
        )

        new_parent_block = block_map[parent_block]
        new_parent_block = block_map[parent_block]
        assert new_parent_block is new_parent_block

        interim_ld = new_parent_block.get_operation_at_index(ld_idx)
        assert isinstance(interim_ld, memref.Load)
        interim_ld.res.replace_by(new_block_arg)
        interim_ld.detach()
        interim_ld.erase()

        st_idx = st_idx - 1

        interim_st = new_parent_block.get_operation_at_index(st_idx)
        assert isinstance(interim_st, memref.Store)
        new_yield_val = interim_st.value
        interim_st.detach()
        interim_st.erase()

        # yield the value that was used in the old store
        assert new_body.block.last_op is not None
        assert new_yield_val is not None
        rewriter.replace_op(new_body.block.last_op, scf.Yield(new_yield_val))

        new_loop = scf.For(
            outer_loop.lb, outer_loop.ub, outer_loop.step, [new_ld], new_body
        )

        # use new loop's yielded result in a store after the loop
        assert len(new_loop.res) == 1
        new_st = memref.Store.get(new_loop.res[0], store_op.memref, store_op.indices)
        rewriter.insert_op_after(new_st, outer_loop)

        rewriter.insert_op_before(new_loop, outer_loop)
        rewriter.erase_op(outer_loop)


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
            apply_recursively=True,
            walk_reverse=True,
        ).rewrite_module(op)
