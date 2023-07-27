from collections import defaultdict
from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.dialects import riscv, riscv_func, riscv_scf
from xdsl.ir import BlockArgument, Region, SSAValue


@Builder.implicit_region([riscv.Registers.A0, riscv.Registers.A1])
def myfunc_body(args: tuple[BlockArgument, ...]):
    a0, a1 = args
    # these moves are added by default by the lowering to func
    lb = riscv.MVOp(a0)
    ub = riscv.MVOp(a1)

    step = riscv.LiOp(1)
    acc = riscv.LiOp(0)

    @Builder.implicit_region(
        [riscv.IntRegisterType.unallocated(), riscv.IntRegisterType.unallocated()]
    )
    def loop_body(args: tuple[BlockArgument, ...]):
        i, acc = args
        new_acc = riscv.AddOp(acc, i)
        riscv_scf.YieldOp(new_acc)

    loop = riscv_scf.ForOp(lb, ub, step, [acc], loop_body)

    a0_new = riscv.MVOp(loop.res[0], rd=riscv.Registers.A0)

    riscv_func.ReturnOp([a0_new])


testfunc = riscv_func.FuncOp("funky", myfunc_body)


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

    def free_reg(self) -> riscv.IntRegisterType:
        for reg in (
            *riscv.Registers.A,
            *riscv.Registers.T,
            *riscv.Registers.S,
        ):
            if reg not in self.forbidden_vals and self.liveliness[reg] == 0:
                return reg
        assert False, "Out of registers"

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
        assert isinstance(arg.type, riscv.IntRegisterType)
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

    # grab a free register for the loop counter (from the inner context)
    loop_counter_reg = inner_ctx.free_reg()

    # if the lb has no other uses, use it as the loop counter
    if len(op.lb.uses) == 1:
        loop_counter_reg = op.lb.type

    iter_val = op.body.block.args[0]
    iter_val.type = loop_counter_reg
    inner_ctx.register_use(iter_val)
    # we don't need to make the iter val forbidden, as we "force" it to be
    # in the right register at the end of the loop, it could meander around in between
    # operations if it wants to.
    inner_ctx.forbidden_vals.remove(loop_counter_reg)

    register_allocate_region(op.body, inner_ctx)


def register_allocate_region(reg: Region, ctx: RegAllocCtx):
    """
    Iterate over all ops in a region and register allocate them
    """
    for block in reg.blocks:
        for op in block.ops:
            # handle the "default" riscv instructions with rd and rs regs
            # this is how we register allocate 99% of riscv operations
            if isinstance(
                op,
                (
                    riscv.RdImmIntegerOperation,
                    riscv.RdRsRsIntegerOperation,
                    riscv.RdRsIntegerOperation,
                    # TODO: add more
                ),
            ):
                # keep track of the "alive" registers
                for operand in op.operands:
                    ctx.register_use(operand)
                # skip already allocated registers
                if op.rd.type.is_allocated:
                    continue
                # grab a free register and set it
                reg = ctx.free_reg()
                op.rd.type = reg
                ctx.add_reg(op.rd)
            # the scf for loop has a special case function
            elif isinstance(op, riscv_scf.ForOp):
                register_allocate_for_op(op, ctx)
            else:
                # unknown ops without results are fine, I think?
                if len(op.results) == 0:
                    continue
                raise RuntimeError(f"Unknown op {op}")


if __name__ == "__main__":
    # print(testfunc)

    register_allocate_function(testfunc)

    print(testfunc)
