from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.dialects.gpu import ModuleEndOp, ModuleOp


def test_gpu_module():
    name = SymbolRefAttr.from_str("gpu")

    ops = [ModuleEndOp.get()]

    gpu_module = ModuleOp.get(name, ops)

    assert gpu_module.body.ops == ops
    assert gpu_module.sym_name is name


def test_gpu_module_end():
    module_end = ModuleEndOp.get()

    assert isinstance(module_end, ModuleEndOp)
