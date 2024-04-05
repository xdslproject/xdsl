from dataclasses import dataclass

from xdsl.dialects import accfg, builtin, scf
from xdsl.ir import MLContext, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.accfg.trace_acc_state import infer_state_of


class SimplifyRedundantSetupCalls(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.SetupOp, rewriter: PatternRewriter, /):
        # Step 1: Figure out previous state
        prev_state = infer_state_of(op.in_state) if op.in_state else {}

        # Step 2: Filter setup parameters to remove calls that set the same value again
        new_params: list[tuple[str, SSAValue]] = [
            (name, val) for name, val in op.iter_params() if prev_state.get(name) != val
        ]

        # Step 3: If no new params remain, elide the whole op
        if not new_params:
            # This only happens when:
            #  1) The operation has no input state, and
            #  2) The operation has no parameters it sets
            if op.in_state is None:
                # in this case, we can't elide the operation if it's output state is
                # used otherwise we would break stuff. So we just assume that a setup
                # without parameters returns an "empty" state that assumes nothing.
                return

            op.out_state.replace_by(op.in_state)
            rewriter.erase_matched_op()
            return

        # Step 4: If all parameters change, do nothing
        if len(new_params) == len(op.param_names):
            return

        # Step 5: Replace matched op with reduced version
        rewriter.replace_matched_op(
            accfg.SetupOp(
                [val for _, val in new_params],
                [param for param, _ in new_params],
                op.accelerator,
                op.in_state,
            )
        )


class HoistSetupCallsIntoConditionals(RewritePattern):
    """
    Hoists setup calls into preceeding scf.if blocks if possible.

    This will never result in additional setup calls and only
    reduced the number of fields that are set up.
    (proof left as an exercise to the reader)

    Things we need to worry about:

    - we can't insert a setup op before the originally produced state.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.SetupOp, rewriter: PatternRewriter, /):
        # do not apply if our in_state is not an scf.if
        if op.in_state is None or not isinstance(op.in_state.owner, scf.If):
            return
        # grab some helper vars
        old_in_state = op.in_state
        assert isinstance(old_in_state, OpResult)

        # Step 1: Check that it's legal to move:
        # grab all launch op uses of the SSA value produced by the scf.if
        # this will only find things that happen *after* the scf.if, so nothing
        # inside the scf.if regions.
        uses = tuple(
            use for use in op.in_state.uses if isinstance(use.operation, accfg.LaunchOp)
        )
        # if we have some launch ops, we need to investigate further:
        for use in uses:
            # grab the launch operation
            launch_op = use.operation
            # check that it is in the same block (if not bail out)
            # if it is not in the same block, it means that the launch is nested:
            #   scf.if -> some_op { launch } -> setup
            # properly inferring where the launch op is, is out of scope for now.
            # we always bail out in this case so that we don't break things.
            # TODO: better check this case?
            if launch_op.parent_block() is not op.parent_block():
                return
            # if we share a block, figure out positions and compare:
            block = launch_op.parent_block()
            assert block is not None
            # launch op index < our index => we have a launch between us and the if
            # that means we can't hoist.
            if block.get_operation_index(launch_op) < block.get_operation_index(op):
                return

        # Step 2: Clone the op into the end of both branches
        for region in op.in_state.owner.regions:
            # grab the yield op:
            yield_op = region.block.last_op
            assert isinstance(yield_op, scf.Yield)
            new_in_state = yield_op.operands[old_in_state.index]

            # insert a copy of the setup op but replace the
            # original in_state with the yielded state of the region
            rewriter.insert_op_before(
                new_setup := op.clone(value_mapper={op.in_state: new_in_state}),
                yield_op,
            )
            # replace the yield op with a new yield op that yields the
            # newly produced state
            rewriter.replace_op(
                yield_op,
                yield_op.clone(value_mapper={new_in_state: new_setup.out_state}),
            )

        # erase the op, replacing all uses of its out state by
        # its in_state.
        op.out_state.replace_by(op.in_state)
        rewriter.erase_matched_op()


@dataclass(frozen=True)
class AccfgDeduplicate(ModulePass):
    """
    Reduce the number of parameters in setup calls by inferring previously
    set up values and carefully moving setup calls around.
    """

    name = "accfg-dedup"

    hoist: bool = True

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        patterns = [
            SimplifyRedundantSetupCalls(),
        ]

        if self.hoist:
            patterns.append(HoistSetupCallsIntoConditionals())

        PatternRewriteWalker(
            GreedyRewritePatternApplier(patterns),
            walk_reverse=True,
        ).rewrite_module(op)
