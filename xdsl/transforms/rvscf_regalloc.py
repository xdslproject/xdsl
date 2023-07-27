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
    vals_to_regs: dict[SSAValue, riscv.RISCVRegisterType] = field(default_factory=dict)
    used_regs: set[riscv.RISCVRegisterType] = field(default_factory=set)

    def clone(self):
        return RegAllocCtx(
            vals_to_regs=self.vals_to_regs.copy(),
            used_regs=self.used_regs.copy(),
        )

    def free_reg(self):
        for reg in [
            riscv.Registers.T0,
            riscv.Registers.T1,
            riscv.Registers.T2,
            riscv.Registers.T3,
            riscv.Registers.T4,
            riscv.Registers.T5,
            # TODO: ...
        ]:
            if reg not in self.used_regs:
                return reg

    def add_reg(self, reg: riscv.RISCVRegisterType, val: SSAValue):
        self.used_regs.add(reg)
        self.vals_to_regs[val] = reg


def register_allocate_function(func: riscv_func.FuncOp):
    ctx = RegAllocCtx()

    for arg in func.func_body.blocks[0].args:
        assert isinstance(arg.type, riscv.IntRegisterType)
        assert arg.type.is_allocated
        ctx.add_reg(arg.type, arg)

    register_allocate_region(func.func_body, ctx)

    ret_op = func.func_body.block.last_op
    assert isinstance(ret_op, riscv_func.ReturnOp)
    # TODO: abstract away "a registers"
    assert len(ret_op.operands) < 4
    for reg, val in zip(
        [riscv.Registers.A0, riscv.Registers.A1, riscv.Registers.A2], ret_op.operands
    ):
        val.type = reg
        ctx.add_reg(reg, val)


def register_allocate_for_op(op: riscv_scf.ForOp, ctx: RegAllocCtx):
    """
    Register allocate the for loop

    Take care of:
        - loop counter stays in same register
        - loop carried variables stay in same register
    """
    yield_op = op.body.block.last_op
    assert isinstance(yield_op, riscv_scf.YieldOp)
    for input_val, block_arg, yielded, ret_val in zip(
        op.iter_args, op.body.block.args[1:], yield_op.operands, op.results
    ):
        block_arg.type = input_val.type
        yielded.type = input_val.type
        ret_val.type = input_val.type

    loop_counter_reg = ctx.free_reg()

    # if the lb has no used, use it as the loop counter
    if len(op.lb.uses) == 1:
        loop_counter_reg = op.lb.type

    iter_val = op.body.block.args[0]
    iter_val.type = loop_counter_reg
    ctx.add_reg(loop_counter_reg, iter_val)

    register_allocate_region(op.body, ctx)


def register_allocate_region(reg: Region, ctx: RegAllocCtx):
    for op in reg.blocks[0].ops:
        if isinstance(op, riscv.RdImmIntegerOperation):
            if op.rd.type.is_allocated:
                continue
            reg = ctx.free_reg()
            op.rd.type = reg
            ctx.add_reg(reg, op.rd)
        elif isinstance(op, riscv.RdRsRsIntegerOperation):
            if op.rd.type.is_allocated:
                continue
            reg = ctx.free_reg()
            op.rd.type = reg
            ctx.add_reg(reg, op.rd)
        elif isinstance(op, riscv_scf.ForOp):
            register_allocate_for_op(op, ctx)
        elif isinstance(op, riscv.MVOp):
            reg = ctx.free_reg()
            if op.rd.type.is_allocated:
                break
            # TODO: this is a hack
            reg = op.rs.type
            op.rd.type = reg
            ctx.add_reg(reg, op.rd)
        else:
            if len(op.results) == 0:
                continue
            raise RuntimeError(f"Unknown op {op}")

if __name__ == '__main__':
    register_allocate_function(testfunc)

    print(testfunc)
