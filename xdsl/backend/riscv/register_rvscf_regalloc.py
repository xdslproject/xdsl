from collections import defaultdict
from dataclasses import dataclass, field

from xdsl.dialects import riscv, riscv_func, riscv_scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class RegAllocCtx:
    """
    This is the context object for allocating registers.

    It keeps track of how many times each ssa value is expected to be seen before
    it is "free" again.

    It also carries a list of forbidden registers, as they are used by the "outside"
    (i.e. parent region) and shall not be overwritten.

    This is currently still a bit broken, as ssa values can have the same register
    even before.
    """

    liveliness: dict[riscv.RISCVRegisterType, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    """
    dictionary mapping registers to how many uses they have "left"
    """

    forbidden_vals: set[riscv.RISCVRegisterType] = field(default_factory=set)
    """
    A set of registers that are forbidden for allocation
    """

    def child_for_loops(self):
        """
        Create a new context that cannot overwrite any registers from the "parent"
        context.

        Moves all registers with remaining uses to forbidden_vals
        """
        return RegAllocCtx(
            forbidden_vals={reg for reg, cnt in self.liveliness.items() if cnt > 0}
        )

    def free_int_reg(self) -> riscv.IntRegisterType:
        for reg in (
            *riscv.Registers.A,
            *riscv.Registers.T,
            *riscv.Registers.S,
        ):
            if reg not in self.forbidden_vals and self.liveliness[reg] == 0:
                return reg
        raise NotImplementedError("Out of registers, spilling not implemented")

    def free_float_reg(self) -> riscv.FloatRegisterType:
        for reg in (
            *riscv.Registers.FA,
            *riscv.Registers.FT,
            *riscv.Registers.FS,
        ):
            if reg not in self.forbidden_vals and self.liveliness[reg] == 0:
                return reg
        raise NotImplementedError("Out of registers, spilling not implemented")

    def add_reg(self, val: SSAValue):
        assert isinstance(val.type, riscv.RISCVRegisterType) and val.type.is_allocated
        if self.liveliness[val.type] > 0:
            # this may be interesting behaviour? idk
            pass
        self.liveliness[val.type] = len(val.uses)

    def register_use(self, val: SSAValue):
        assert isinstance(val.type, riscv.RISCVRegisterType) and val.type.is_allocated
        if self.liveliness[val.type] > 0:
            self.liveliness[val.type] -= 1

    def make_forbidden(self, reg: riscv.RISCVRegisterType):
        self.forbidden_vals.add(reg)


def register_allocate_function(func: riscv_func.FuncOp):
    # create an empty context
    ctx = RegAllocCtx()

    # register all function args and their liveliness
    for arg in func.func_body.blocks[0].args:
        assert isinstance(arg.type, riscv.RISCVRegisterType)
        assert arg.type.is_allocated
        ctx.add_reg(arg)

    # allocate the body
    register_allocate_region(func.func_body, ctx)


def register_allocate_for_op(op: riscv_scf.ForOp, ctx: RegAllocCtx):
    """
    Register allocate the for loop

    Take care of:
        - loop counter stays in same register
        - loop carried variables stay in same register
    """
    # construct an inner contex for the loop body
    inner_ctx = ctx.child_for_loops()

    # make sure all loop carried variables stay in the same registers during the loop:
    # gently force the block argument and yielded values in the same register as the
    # given values
    yield_op = op.body.block.last_op
    assert isinstance(yield_op, riscv_scf.YieldOp)
    for input_val, block_arg, yielded, ret_val in zip(
        op.iter_args, op.body.block.args[1:], yield_op.operands, op.results
    ):
        block_arg.type = input_val.type
        yielded.type = input_val.type
        ret_val.type = input_val.type

    # if the lb has no other uses, use it as the loop counter otherwise
    # grab a free register for the loop counter (from the inner context)
    loop_counter_reg = op.lb.type if len(op.lb.uses) == 1 else inner_ctx.free_int_reg()
    assert isinstance(loop_counter_reg, riscv.IntRegisterType)

    iter_val = op.body.block.args[0]
    iter_val.type = loop_counter_reg
    inner_ctx.register_use(iter_val)
    # make the iter_val forbidden so it is not overwritten in the loop
    inner_ctx.make_forbidden(loop_counter_reg)

    register_allocate_region(op.body, inner_ctx)


def register_allocate_region(reg: Region, ctx: RegAllocCtx):
    """
    Iterate over all ops in a region and register allocate them
    """
    for block in reg.blocks:
        for op in block.ops:
            if isinstance(
                op,
                (riscv.RISCVOp,),
            ):
                for result in op.results:
                    assert isinstance(result.type, riscv.RISCVRegisterType)
                    # keep track of the "alive" registers
                    for operand in op.operands:
                        ctx.register_use(operand)
                    # skip already allocated registers
                    if result.type.is_allocated:
                        continue
                    # grab a free register and set it
                    result.type = (
                        ctx.free_int_reg()
                        if isinstance(result.type, riscv.IntRegisterType)
                        else ctx.free_float_reg()
                    )
                    ctx.add_reg(result)
            # the scf for loop has a special case function
            elif isinstance(op, riscv_scf.ForOp):
                register_allocate_for_op(op, ctx)
            else:
                if len(op.results) == 0:
                    continue
                raise ValueError(
                    f"SCF Register allocation for {op}"
                    f"with {op.results} is not implemented"
                )


class AllocateRISCVFunction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_func.FuncOp, _: PatternRewriter) -> None:
        register_allocate_function(op)


@dataclass
class RVSCFRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers for all riscv functions in a module
    """

    name = "rvscf-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AllocateRISCVFunction(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
