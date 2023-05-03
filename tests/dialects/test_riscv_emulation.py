from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import riscv


def test_simple():
    @ModuleOp
    @Builder.implicit_region
    def module():
        six = riscv.LiOp(6).rd
        five = riscv.LiOp(5).rd
        riscv.AddOp(six, five).rd

    riscv_code = riscv.riscv_code(module)
    print(module)
    print(riscv_code)
