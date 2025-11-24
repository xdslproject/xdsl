from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_cf, riscv_scf
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import BlockInsertPoint, InsertPoint


class LowerRiscvScfForPattern(RewritePattern):
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
         |   riscv_cf.bge %iv, %ub, end, body (%iv, %args_init...)      |
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
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter, /):
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

        # The first argument of the loop body block is the loop counter by SCF invariant.
        loop_var_reg = body.args[0].type
        assert isinstance(loop_var_reg, riscv.IntRegisterType)

        # Use the first block of the loop body as the condition block since it is the
        # block that has the induction variable and loop-carried values as arguments.
        # Split out all operations from the first block into a new block. Move all
        # body blocks from the loop body region to the region containing the loop.
        first_body_block = op.body.blocks[0]
        last_body_block = op.body.blocks[-1]

        # Get the induction variable and its register
        iv = first_body_block.args[0]
        iv_reg = iv.type
        assert isinstance(iv_reg, riscv.IntRegisterType)

        # Append the induction variable stepping logic to the last body block, add
        # comparison with upper bound, and conditionally branch back into the body.
        yield_op = last_body_block.last_op
        assert isinstance(yield_op, riscv_scf.YieldOp)

        rewriter.replace_op(
            yield_op,
            (
                add_op := riscv.AddOp(iv, op.step, rd=iv_reg),
                riscv_cf.BltOp(
                    add_op.rd,
                    op.ub,
                    (add_op.rd, *yield_op.operands),
                    (add_op.rd, *yield_op.operands),
                    first_body_block,
                    end_block,
                ),
            ),
        )

        rewriter.inline_region(op.body, BlockInsertPoint.before(end_block))

        # Move lb to new register to initialize the iv.
        # Skip for loop if condition is not satisfied at start.
        rewriter.insert_op(
            (
                mv_op := riscv.MVOp(op.lb, rd=iv_reg),
                riscv_cf.BgeOp(
                    mv_op.rd,
                    op.ub,
                    (mv_op.rd, *op.iter_args),
                    (mv_op.rd, *op.iter_args),
                    end_block,
                    first_body_block,
                ),
            ),
            InsertPoint.at_end(init_block),
        )

        # Insert label at the start of the first body block.
        rewriter.insert_op(
            riscv.LabelOp(f"scf_body_{suffix}"), InsertPoint.at_start(first_body_block)
        )

        # Replace operation by arguments to the newly end block.
        rewriter.replace_op(
            op,
            riscv.LabelOp(f"scf_body_end_{suffix}"),
            end_block.args[1:],
        )


class ConvertRiscvScfToRiscvCfPass(ModulePass):
    name = "convert-riscv-scf-to-riscv-cf"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LowerRiscvScfForPattern(), walk_regions_first=True
        ).rewrite_module(op)
