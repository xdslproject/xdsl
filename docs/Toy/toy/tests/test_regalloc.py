from xdsl.builder import Builder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.riscv_asm_writer import riscv_code

from xdsl.transforms.riscv_register_allocation import (
    RISCVRegisterAllocation,
)


def context() -> MLContext:
    ctx = MLContext()
    return ctx


# Handwritten riscv dialect code to test register allocation


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


@ModuleOp
@Builder.implicit_region
def simple_linear_riscv_allocated():
    """
    Register allocated version based on BlockNaive strategy of the code in simple_linear_riscv.
    """

    @Builder.implicit_region
    def text_region():
        @Builder.implicit_region
        def main_region() -> None:
            zero = riscv.GetRegisterOp(riscv.Registers.ZERO).res
            v0 = riscv.AddiOp(zero, 1, rd=riscv.Registers.T6).rd
            v1 = riscv.AddiOp(zero, 2, rd=riscv.Registers.T5).rd

            v3 = riscv.AddOp(v1, v0, rd=riscv.Registers.T4).rd
            v4 = riscv.MulOp(v3, v1, rd=riscv.Registers.T3).rd
            v5 = riscv.SubOp(v0, v4, rd=riscv.Registers.S11).rd
            v6 = riscv.MulOp(v1, v0, rd=riscv.Registers.S10).rd
            v7 = riscv.AddOp(v6, v3, rd=riscv.Registers.S9).rd
            v8 = riscv.AddOp(v7, v5, rd=riscv.Registers.S8).rd
            v9 = riscv.SubOp(v0, v1, rd=riscv.Registers.S7).rd
            v10 = riscv.MulOp(v5, v3, rd=riscv.Registers.S6).rd
            v11 = riscv.AddOp(v9, v10, rd=riscv.Registers.S5).rd
            v12 = riscv.AddOp(v11, v8, rd=riscv.Registers.S4).rd
            v13 = riscv.AddOp(v4, v0, rd=riscv.Registers.S3).rd
            v14 = riscv.SubOp(v13, v5, rd=riscv.Registers.S2).rd
            v15 = riscv.MulOp(v12, v8, rd=riscv.Registers.A7).rd
            v16 = riscv.AddOp(v14, v15, rd=riscv.Registers.A6).rd

            riscv.MVOp(v16, rd=riscv.Registers.A0)
            riscv.AddiOp(zero, 93, rd=riscv.Registers.A7).rd
            riscv.EcallOp()

        riscv.LabelOp("main", main_region)

    riscv.DirectiveOp(".text", None, text_region)


def test_allocate_simple_linear():
    RISCVRegisterAllocation("BlockNaive").apply(context(), simple_linear_riscv)

    assert riscv_code(simple_linear_riscv) == riscv_code(simple_linear_riscv_allocated)
