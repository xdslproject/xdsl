from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.riscv import AssemblySectionOp, DirectiveOp, LabelOp
from xdsl.ir import Block, Region

from ..rewrites.setup_riscv_pass import SetupRiscvPass

with ImplicitBuilder((input_module := ModuleOp([])).body):
    FuncOp("main", ((), ()))

with ImplicitBuilder((output_module := ModuleOp([])).body):
    bss = AssemblySectionOp(".bss", Region(Block()))
    with ImplicitBuilder(bss.data):
        LabelOp("heap")
        DirectiveOp(".space", f"{1024}")
    data = AssemblySectionOp(".data", Region(Block()))
    FuncOp("main", ((), ()))


def test_add_sections():
    module = input_module.clone()
    SetupRiscvPass().apply(MLContext(), module)
    assert f"{module}" == f"{output_module}"
