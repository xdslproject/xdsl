from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.builder import Builder
from xdsl.dialects import riscv, riscv_func
from xdsl.dialects.builtin import SymbolRefAttr


def test_gather_allocated():
    @Builder.implicit_region
    def no_preallocated_body() -> None:
        v1 = riscv.GetRegisterOp(riscv.Registers.UNALLOCATED_INT).res
        v2 = riscv.GetRegisterOp(riscv.Registers.UNALLOCATED_INT).res
        _ = riscv.AddOp(v1, v2).rd

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(no_preallocated_body)
    )

    assert pa_regs == set()

    @Builder.implicit_region
    def one_preallocated_body() -> None:
        reg1 = riscv.Registers.UNALLOCATED_INT
        v1 = riscv.GetRegisterOp(reg1).res
        v2 = riscv.GetRegisterOp(riscv.Registers.A7).res
        _ = riscv.AddOp(v1, v2).rd

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(one_preallocated_body)
    )

    assert pa_regs == {riscv.Registers.A7}

    @Builder.implicit_region
    def repeated_preallocated_body() -> None:
        v1 = riscv.GetRegisterOp(riscv.Registers.UNALLOCATED_INT).res
        v2 = riscv.GetRegisterOp(riscv.Registers.A7).res
        sum1 = riscv.AddOp(v1, v2).rd
        _ = riscv.AddiOp(sum1, 1, rd=riscv.Registers.A7).rd

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(repeated_preallocated_body)
    )

    assert pa_regs == {riscv.Registers.A7}

    @Builder.implicit_region
    def multiple_preallocated_body() -> None:
        v1 = riscv.GetRegisterOp(riscv.Registers.UNALLOCATED_INT).res
        v2 = riscv.GetRegisterOp(riscv.Registers.A7).res
        sum1 = riscv.AddOp(v1, v2).rd
        _ = riscv.AddiOp(sum1, 1, rd=riscv.Registers.A6).rd

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(multiple_preallocated_body)
    )

    assert pa_regs == {riscv.Registers.A6, riscv.Registers.A7}

    @Builder.implicit_region
    def func_call_preallocated_body() -> None:
        v1 = riscv.GetRegisterOp(riscv.Registers.UNALLOCATED_INT).res
        v2 = riscv.GetRegisterOp(riscv.Registers.S0).res
        riscv_func.CallOp(SymbolRefAttr("hello"), (v1, v2), ())

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(
            func_call_preallocated_body
        )
    )

    assert len(pa_regs) == 36
    assert riscv.Registers.S0 in pa_regs
