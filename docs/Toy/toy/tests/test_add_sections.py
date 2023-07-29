from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.riscv import DirectiveOp, LabelOp
from xdsl.ir import Block, MLContext, Region

from ..rewrites.setup_riscv_pass import SetupRiscvPass

with ImplicitBuilder((input_module := ModuleOp([])).body):
    FuncOp("main", ((), ()))

with ImplicitBuilder((output_module := ModuleOp([])).body):
    bss = DirectiveOp(".bss", None, Region(Block()))
    with ImplicitBuilder(bss.data):
        LabelOp("heap")
        DirectiveOp(".space", f"{1024}")
    data = DirectiveOp(".data", None, Region(Block()))
    text = DirectiveOp(".text", None, Region(Block()))
    with ImplicitBuilder(text.data):
        FuncOp("main", ((), ()))


def test_add_sections():
    module = input_module.clone()
    SetupRiscvPass().apply(MLContext(), module)
    assert f"{module}" == f"{output_module}"
