from xdsl.backend.riscv.register_allocation import gather_allocated
from xdsl.builder import Builder
from xdsl.dialects import riscv, riscv_func


def test_gather_allocated():
    @Builder.implicit_region
    def no_preallocated_body() -> None:
        reg1 = riscv.IntRegisterType.unallocated()
        reg2 = riscv.IntRegisterType.unallocated()
        v1 = riscv.GetRegisterOp(reg1).res
        v2 = riscv.GetRegisterOp(reg2).res
        _ = riscv.AddOp(v1, v2, rd=riscv.IntRegisterType.unallocated()).rd

    pa_regs = gather_allocated(riscv_func.FuncOp("foo", no_preallocated_body, ((), ())))

    assert len(pa_regs) == 0

    @Builder.implicit_region
    def one_preallocated_body() -> None:
        reg1 = riscv.IntRegisterType.unallocated()
        v1 = riscv.GetRegisterOp(reg1).res
        v2 = riscv.GetRegisterOp(riscv.Registers.A7).res
        _ = riscv.AddOp(v1, v2, rd=riscv.IntRegisterType.unallocated()).rd

    pa_regs = gather_allocated(
        riscv_func.FuncOp("foo", one_preallocated_body, ((), ()))
    )

    assert len(pa_regs) == 1

    @Builder.implicit_region
    def repeated_preallocated_body() -> None:
        reg1 = riscv.IntRegisterType.unallocated()
        v1 = riscv.GetRegisterOp(reg1).res
        v2 = riscv.GetRegisterOp(riscv.Registers.A7).res
        sum1 = riscv.AddOp(v1, v2, rd=riscv.IntRegisterType.unallocated()).rd
        _ = riscv.AddiOp(sum1, 1, rd=riscv.Registers.A7).rd

    pa_regs = gather_allocated(
        riscv_func.FuncOp("foo", repeated_preallocated_body, ((), ()))
    )

    assert len(pa_regs) == 1

    @Builder.implicit_region
    def multiple_preallocated_body() -> None:
        reg1 = riscv.IntRegisterType.unallocated()
        v1 = riscv.GetRegisterOp(reg1).res
        v2 = riscv.GetRegisterOp(riscv.Registers.A7).res
        sum1 = riscv.AddOp(v1, v2, rd=riscv.IntRegisterType.unallocated()).rd
        _ = riscv.AddiOp(sum1, 1, rd=riscv.Registers.A6).rd

    pa_regs = gather_allocated(
        riscv_func.FuncOp("foo", multiple_preallocated_body, ((), ()))
    )

    assert len(pa_regs) == 2
