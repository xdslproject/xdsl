from xdsl.context import Context
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.dialects.x86.registers import RFLAGS, GeneralRegisterType
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import BlockInsertPoint, InsertPoint


class LowerX86ScfForPattern(RewritePattern):
    """
    Inline the for loop body into its parent region, using `Block`s to represent control
    flow. The `Block` containing the `ForOp` is split into two, and the blocks in the
    `body` of the for loop are spliced between them. Additional operations are inserted
    into the block before, and the block after to handle the initialization of the
    iteration argument, and loop-carried variables, as well as control flow. If the for
    loop contained other `riscv_scf` ops, they will have been rewritten by the time this
    rewrite is called. Two comparison operations are inserted, one just before the loop
    blocks, skipping the loop entirely if the condition is not met, and one at the end of
    the loop body, to exit or continue the loop. A canonicalization step may be able to
    eliminate the first check if the bounds are known at compile time.
    ```

         +--------------------------------------------------------------+
         |   <code before the ForOp>                                    |
         |   <definitions of %args_init...>                             |
         |   <compute initial %iv value>                                |
         |   cmp %iv, %ub                                               |
         |   x86.jge end, body (%iv, %args_init...)                     |
         +--------------------------------------------------------------+
                                     |               |
     -------------------|            |               -----------------------|
     |                  v            v                                      |
     |   +--------------------------------------------------------------+   |
     |   | body-first(%iv, %args_body...):                              |   |
     |   |   <body contents>                                            |   |
     |   +--------------------------------------------------------------+   |
     |                               |                                      |
     |                              ...                                     |
     |                               |                                      |
     |   +--------------------------------------------------------------+   |
     |   | body-last:                                                   |   |
     |   |   <body contents>                                            |   |
     |   |   <%yields... = operands of yield>                           |   |
     |   |   <%ub and %step visible by dominance>                       |   |
     |   |   %new_iv =<add %step to %iv>                                |   |
     |   |   riscv_cf.blt %new_iv, %ub, body, end (%new_iv, %yields...) |   |
     |   +--------------------------------------------------------------+   |
     |                 |             |                                      |
     |------------------             |               |-----------------------
                                     v               v
         +--------------------------------------------------------------+
         | end(%iv, %args_end...):                                      |
         |   <results of ForOp = %args_end>                             |
         |   <code after the ForOp>                                     |
         +--------------------------------------------------------------+
    ```
    """

    for_idx: int

    def __init__(self):
        super().__init__()
        self.for_idx = -1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: x86_scf.ForOp, rewriter: PatternRewriter, /):
        # To ensure that we have a unique labels for each (nested) loop, we use an index
        # that is incremented for each loop as a suffix.
        self.for_idx += 1
        suffix = f"{self.for_idx}_for"

        # Start by splitting the block containing the 'scf.for' into two parts.
        # The part before will get the init code, the part after will be the end point.

        init_block = op.parent_block()
        assert init_block is not None

        body = op.body.blocks[0]

        # TODO: add method to rewriter
        end_block = init_block.split_before(op, arg_types=body.arg_types)

        # Use the first block of the loop body as the condition block since it is the
        # block that has the induction variable and loop-carried values as arguments.
        # Split out all operations from the first block into a new block. Move all
        # body blocks from the loop body region to the region containing the loop.
        first_body_block = op.body.blocks[0]
        last_body_block = op.body.blocks[-1]

        # Get the induction variable and its register
        iv_body = SSAValue.get(first_body_block.args[0], type=GeneralRegisterType)
        iv_used = iv_body.first_use is not None
        iv_reg = iv_body.type
        ub = op.ub
        step = op.step

        # Insert label at the start of the first body block.
        rewriter.insert_op(
            body_label_op := x86.ops.LabelOp(f"scf_body_{suffix}"),
            InsertPoint.at_start(first_body_block),
        )

        # Append the induction variable stepping logic to the last body block, add
        # comparison with upper bound, and conditionally branch back into the body.
        yield_op = last_body_block.last_op
        assert isinstance(yield_op, x86_scf.YieldOp)

        match step:
            case SSAValue():
                step_op = x86.ops.RS_AddOp(iv_body, step)
            case builtin.IntegerAttr():
                step_immediate = step
                if step.value.data == 1:
                    step_op = x86.ops.R_IncOp(iv_body)
                else:
                    step_op = x86.ops.RI_AddOp(
                        iv_body,
                        step_immediate,
                    )
        step_op.register_out.name_hint = iv_body.name_hint
        new_iv = step_op.register_out

        match ub:
            case SSAValue():
                cmp_op = x86.ops.SS_CmpOp(new_iv, ub, result=RFLAGS)
            case builtin.IntegerAttr():
                cmp_op = x86.ops.SI_CmpOp(new_iv, ub)

        # Insert comparison and jump to beginning of loop
        rewriter.replace_op(
            yield_op,
            (
                cmp_op,
                x86.ops.C_JlOp(
                    cmp_op.result,
                    (new_iv, *yield_op.operands),
                    (new_iv, *yield_op.operands),
                    first_body_block,
                    end_block,
                ),
            ),
        )

        # Insert iv increment
        # If iv was not used prior to lowering, then put it at the start of the loop as
        # an optimisation to avoid cycles waiting for the increment.
        rewriter.insert_op(
            step_op,
            InsertPoint.before(cmp_op) if iv_used else InsertPoint.after(body_label_op),
        )

        end_block.args[0].name_hint = iv_body.name_hint

        rewriter.inline_region(op.body, BlockInsertPoint.before(end_block))

        # Move lb to new register to initialize the iv.
        if op.lb.type == iv_reg:
            # Just use lb for iv, no need to move to self
            iv_condition = op.lb
        else:
            iv_condition = rewriter.insert_op(
                x86.ops.DS_MovOp(op.lb, destination=iv_reg),
                InsertPoint.at_end(init_block),
            ).destination
            iv_condition.name_hint = op.lb.name_hint

        if (
            isinstance(lb_owner := op.lb.owner, Operation)
            and isinstance(lb_owner, x86.DI_MovOp)
            and isinstance(ub, builtin.IntegerAttr)
            and lb_owner.immediate.value.data < ub.value.data
        ):
            # Loop executes at least once, fallthrough directly into it without runtime checks
            rewriter.insert_op(
                (
                    x86.ops.FallthroughOp(
                        (iv_condition, *op.iter_args),
                        first_body_block,
                    ),
                ),
                InsertPoint.at_end(init_block),
            )

            # Replace operation by arguments to the newly added end block.
            rewriter.replace_op(
                op,
                (),
                end_block.args[1:],
            )
        else:
            # Skip for loop if condition is not satisfied at start.
            rewriter.insert_op(
                (
                    cmp_op := (
                        x86.ops.SS_CmpOp(iv_condition, ub, result=RFLAGS)
                        if isinstance(ub, SSAValue)
                        else x86.ops.SI_CmpOp(iv_condition, ub)
                    ),
                    x86.ops.C_JgeOp(
                        cmp_op.result,
                        (iv_condition, *op.iter_args),
                        (iv_condition, *op.iter_args),
                        end_block,
                        first_body_block,
                    ),
                ),
                InsertPoint.at_end(init_block),
            )

            # Replace operation by arguments to the newly added end block.
            rewriter.replace_op(
                op,
                x86.ops.LabelOp(f"scf_body_end_{suffix}"),
                end_block.args[1:],
            )


class ConvertX86ScfToX86Pass(ModulePass):
    name = "convert-x86-scf-to-x86"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LowerX86ScfForPattern(), walk_regions_first=True
        ).rewrite_module(op)
