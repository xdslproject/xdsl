from xdsl.builder import Builder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.riscv_asm_writer import riscv_code

from xdsl.transforms.riscv_register_allocation import (
    RISCVRegisterAllocation,
    RegisterAllocationAlgorithm,
)

from ..emulator.emulator_iop import run_riscv

ALLOCATION_STRATEGIES = [
    RegisterAllocationAlgorithm.GlobalJRegs,
    RegisterAllocationAlgorithm.BlockNaive,
]


def context() -> MLContext:
    ctx = MLContext()
    return ctx


# Handwritten riscv dialect code to test register allocation


@ModuleOp
@Builder.implicit_region
def simple_branching_riscv():
    """
    The following riscv dialect IR is generated from the following C code:

    int main() {
        int a = 5;
        int b = 77777;
        int c = 6;
        if (a==b) {
            c = a * a;
        } else {
            c = b + b;
        }

        return c;
    }

    The goal of this test is to check that the register allocator is able to handle very simple branching code with multiple basic blocks.
    Morever it uses some reserved registers (ra, s0) to check that the register allocator does not use them.
    """

    @Builder.implicit_region
    def text_region():
        @Builder.implicit_region
        def main_region() -> None:
            sp = riscv.GetRegisterOp(riscv.Registers.SP).res
            riscv.AddiOp(sp, -32, rd=riscv.Registers.SP).rd
            ra = riscv.GetRegisterOp(riscv.Registers.RA).res
            riscv.SwOp(ra, sp, 28)
            s0 = riscv.GetRegisterOp(riscv.Registers.S0).res
            riscv.SwOp(s0, sp, 24)
            a = riscv.AddiOp(sp, 32).rd
            b = riscv.LiOp(5).rd
            riscv.SwOp(b, a, -12)
            c = riscv.LuiOp(19).rd
            d = riscv.AddiOp(c, -47).rd
            riscv.SwOp(d, a, -16)
            e = riscv.LiOp(6).rd
            riscv.SwOp(e, a, -20)
            f = riscv.LwOp(a, -12).rd
            g = riscv.LwOp(a, -16).rd
            riscv.BneOp(f, g, riscv.LabelAttr("LBB0_2"))
            riscv.JOp(riscv.LabelAttr("LBB0_1"))

            @Builder.implicit_region
            def true_branch() -> None:
                f = riscv.LwOp(a, -12).rd
                f = riscv.MulOp(f, f).rd
                riscv.SwOp(f, a, -20)
                riscv.JOp(riscv.LabelAttr("LBB0_3"))

            riscv.LabelOp("LBB0_1", true_branch)

            @Builder.implicit_region
            def false_branch() -> None:
                f = riscv.LwOp(a, -16).rd
                f = riscv.AddOp(f, f).rd
                riscv.SwOp(f, a, -20)
                riscv.JOp(riscv.LabelAttr("LBB0_3"))

            riscv.LabelOp("LBB0_2", false_branch)

            @Builder.implicit_region
            def merge_if() -> None:
                f = riscv.LwOp(a, -20).rd
                riscv.LwOp(sp, 28, rd=riscv.Registers.RA).rd
                riscv.LwOp(sp, 24, rd=riscv.Registers.S0).rd
                riscv.AddiOp(sp, 32, rd=riscv.Registers.SP).rd
                riscv.MVOp(f, rd=riscv.Registers.A0)
                zero = riscv.GetRegisterOp(riscv.Registers.ZERO).res
                riscv.AddiOp(zero, 93, rd=riscv.Registers.A7).rd
                riscv.EcallOp()

            riscv.LabelOp("LBB0_3", merge_if)

        riscv.LabelOp("main", main_region)

    riscv.DirectiveOp(".text", None, text_region)


@ModuleOp
@Builder.implicit_region
def simple_linear_riscv():
    """
    The following riscv dialect IR is generated from the following C code:

    int main() {
        int a = 1;
        int b = 2;
        int c = a + b;
        int d = a - b * c;
        int f = a * b + c + d;
        int g = a - b + c * d + f;
        int h = a * b * c * d * f * g;
        return a + b * c - d + f * g;
    }

    The goal of this test is to check that the register allocator is able to handle very simple linear code with no branching.
    """

    @Builder.implicit_region
    def text_region():
        @Builder.implicit_region
        def main_region() -> None:
            zero = riscv.GetRegisterOp(riscv.Registers.ZERO).res
            v0 = riscv.AddiOp(zero, 1).rd
            v1 = riscv.AddiOp(zero, 2).rd

            v3 = riscv.AddOp(v1, v0).rd
            v4 = riscv.MulOp(v3, v1).rd
            v5 = riscv.SubOp(v0, v4).rd
            v6 = riscv.MulOp(v1, v0).rd
            v7 = riscv.AddOp(v6, v3).rd
            v8 = riscv.AddOp(v7, v5).rd
            v9 = riscv.SubOp(v0, v1).rd
            v10 = riscv.MulOp(v5, v3).rd
            v11 = riscv.AddOp(v9, v10).rd
            v12 = riscv.AddOp(v11, v8).rd
            v13 = riscv.AddOp(v4, v0).rd
            v14 = riscv.SubOp(v13, v5).rd
            v15 = riscv.MulOp(v12, v8).rd
            v16 = riscv.AddOp(v14, v15).rd

            riscv.MVOp(v16, rd=riscv.Registers.A0)
            riscv.AddiOp(zero, 93, rd=riscv.Registers.A7).rd
            riscv.EcallOp()

        riscv.LabelOp("main", main_region)

    riscv.DirectiveOp(".text", None, text_region)


def test_allocate_simple_branching():
    for allocation_strategy in ALLOCATION_STRATEGIES:
        RISCVRegisterAllocation(allocation_strategy).apply(
            context(), simple_branching_riscv
        )
        code = riscv_code(simple_branching_riscv)
        assert (
            run_riscv(code, unlimited_regs=True, setup_stack=True, verbosity=1)
            == 155554
        )


def test_allocate_simple_linear():
    for allocation_strategy in ALLOCATION_STRATEGIES:
        RISCVRegisterAllocation(allocation_strategy).apply(
            context(), simple_linear_riscv
        )
        code = riscv_code(simple_linear_riscv)
        assert run_riscv(code, unlimited_regs=True, setup_stack=True, verbosity=1) == 12
