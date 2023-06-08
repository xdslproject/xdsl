from io import StringIO
from typing import List
from xdsl.builder import Builder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext

from xdsl.transforms.riscv_register_allocation import (
    LiveInterval,
    RISCVRegisterAllocation,
    RegisterSet,
)
from xdsl.utils.test_value import TestSSAValue


def context() -> MLContext:
    ctx = MLContext()
    return ctx


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    riscv.print_assembly(module, stream)
    return stream.getvalue()


# Handwritten riscv dialect code to test register allocation
RESERVERD_REGISTERS = set(["zero", "sp", "gp", "tp", "fp", "s0"])
AVAILABLE_REGISTERS = RegisterSet(
    [
        reg
        for reg in list(riscv.Register.ABI_INDEX_BY_NAME.keys())
        if reg not in RESERVERD_REGISTERS
    ]
)


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
    linear = simple_linear_riscv.clone()
    RISCVRegisterAllocation("BlockNaive").apply(context(), linear)

    assert riscv_code(linear) == riscv_code(simple_linear_riscv_allocated)


# Check that the linear scan register allocator is able to allocate given hardcodded live ranges


def test_dummy_linear_scan_allocator():
    def get_test_variable() -> TestSSAValue:
        return TestSSAValue(riscv.RegisterType(riscv.Register()))

    a = LiveInterval(get_test_variable(), 1, 10)
    b = LiveInterval(get_test_variable(), 1, 4)
    c = LiveInterval(get_test_variable(), 1, 3)
    d = LiveInterval(get_test_variable(), 2, 8)
    e = LiveInterval(get_test_variable(), 3, 6)
    f = LiveInterval(get_test_variable(), 3, 10)
    g = LiveInterval(get_test_variable(), 4, 8)

    """
    a: 1 2 3 4 5 6 7 8 9 10
    b: 1 2 3 4
    c: 1 2 3
    d:   2 3 4 5 6 7 8
    e:     3 4 5 6
    f:     3 4 5 6 7 8 9 10
    g:       4 5 6 7 8
    """

    intervals: List[LiveInterval] = [a, b, c, d, e, f, g]

    @ModuleOp
    @Builder.implicit_region
    def empty_module():
        pass

    # create the condition for spilling

    AVAILABLE_REGISTERS.reset()
    AVAILABLE_REGISTERS.limit_free_registers(3)

    RISCVRegisterAllocation("LinearScan").apply(
        context(), empty_module, intervals, AVAILABLE_REGISTERS
    )

    assert a.register == None
    assert a.abstract_stack_location == 0
    assert b.register == riscv.Registers.T0
    assert c.register == riscv.Registers.RA
    assert d.register == None
    assert d.abstract_stack_location == 1
    assert e.register == riscv.Registers.T1
    assert f.register == None
    assert f.abstract_stack_location == 2
    assert g.register == riscv.Registers.RA

    assert a.allocated() == riscv.RegisterType(riscv.Register("j0"))
    assert b.allocated() == riscv.RegisterType(riscv.Registers.T0)
    assert c.allocated() == riscv.RegisterType(riscv.Registers.RA)
    assert d.allocated() == riscv.RegisterType(riscv.Register("j1"))
    assert e.allocated() == riscv.RegisterType(riscv.Registers.T1)
    assert f.allocated() == riscv.RegisterType(riscv.Register("j2"))
    assert g.allocated() == riscv.RegisterType(riscv.Registers.RA)


def test_linear_scan_simple_linear():
    AVAILABLE_REGISTERS.reset()
    AVAILABLE_REGISTERS.limit_free_registers(3)

    linear = simple_linear_riscv.clone()
    RISCVRegisterAllocation("LinearScan").apply(context(), linear, None, AVAILABLE_REGISTERS)

    # TO:DO: check that the registers are allocated correctly
    assert True

        




