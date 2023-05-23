from io import StringIO
from xdsl.builder import Builder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.riscv_asm_writer import riscv_code

from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation, RegisterAllocationType

from ..compiler import (
    compile,
    context,
)
from ..emulator.emulator_iop import run_riscv
from ..emulator.toy_accelerator import ToyAccelerator

# End-to-end test for register allocation
ctx = context()

toy_program = """
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<3, 2> = [1, 2, 3, 4, 5, 6];

  # There is a built-in print instruction to display the contents of the tensor
  print(b);

  # Reshapes are implicit on assignment
  var c<2, 3> = b;

  # There are + and * operators for pointwise addition and multiplication
  var d = a + c;

  print(d);
}
"""

def test_compile_infinite_registers():
    code = compile(toy_program, RegisterAllocationType.BlockNaiveSSA)
    print(code)
    stream = StringIO()
    ToyAccelerator.stream = stream
    run_riscv(code, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=1)
    assert "[[2, 4, 6], [8, 10, 12]]" in stream.getvalue()


# Handwritten riscv dialect code to test register allocation

@ModuleOp
@Builder.implicit_region
def risc_1():
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

    The goal of this test is to check that the register allocator is able to handle very simple branching code.

    """

    @Builder.implicit_region
    def text_region():
        riscv.LabelOp("main")
        sp = riscv.GetRegisterOp(riscv.Registers.SP).res
        a = riscv.AddiOp(sp, -32, rd=riscv.Registers.SP).rd
        ra = riscv.GetRegisterOp(riscv.Registers.RA).res
        riscv.SwOp(ra, a, 28)
        b = riscv.GetRegisterOp(riscv.Registers.S0).res
        riscv.SwOp(b, a, 24)
        b = riscv.AddiOp(sp, 32, rd=riscv.Registers.S0).rd
        c = riscv.LiOp(5).rd
        riscv.SwOp(c, b, -12)
        d = riscv.LuiOp(19).rd
        z = riscv.AddiOp(d, -47).rd
        riscv.SwOp(z, b, -16)
        y = riscv.LiOp(6).rd
        riscv.SwOp(y, b, -20)
        x = riscv.LwOp(b, -12).rd
        a1 = riscv.LwOp(b, -16).rd
        riscv.BneOp(x, a1, riscv.LabelAttr("LBB0_2"))
        riscv.JOp(riscv.LabelAttr("LBB0_1"))
        riscv.LabelOp("LBB0_1")
        x = riscv.LwOp(b, -12).rd
        x = riscv.MulOp(x, x).rd
        riscv.SwOp(x, b, -20)
        riscv.JOp(riscv.LabelAttr("LBB0_3"))
        riscv.LabelOp("LBB0_2")
        x = riscv.LwOp(b, -16).rd
        x = riscv.AddOp(x, x).rd
        riscv.SwOp(x, b, -20)
        riscv.JOp(riscv.LabelAttr("LBB0_3"))
        riscv.LabelOp("LBB0_3")
        x = riscv.LwOp(b, -20).rd
        ra = riscv.LwOp(sp, 28, rd=riscv.Registers.RA).rd
        b = riscv.LwOp(sp, 24, rd=riscv.Registers.S0).rd
        sp = riscv.AddiOp(sp, 32, rd=riscv.Registers.SP).rd
        riscv.MVOp(x, rd=riscv.Registers.A0)
        zero = riscv.GetRegisterOp(riscv.Registers.ZERO).res
        riscv.AddiOp(zero, 93, rd=riscv.Registers.A7).rd
        riscv.CustomEmulatorInstructionOp("scall", (), ())

    riscv.DirectiveOp(".text", None, text_region)


def test_riscv1():
    RISCVRegisterAllocation(RegisterAllocationType.BlockNaiveSSA).apply(ctx, risc_1)
    code = riscv_code(risc_1)
    assert run_riscv(code, unlimited_regs=True, verbosity=1) == 155554


