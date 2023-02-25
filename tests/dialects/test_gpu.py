from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.dialects.gpu import ModuleEndOp, ModuleOp
from xdsl.ir import Region


def test_gpu_module():
    name = SymbolRefAttr.from_str("gpu")

    region = Region.get([])

    gpu_module = ModuleOp.get(name, region)

    assert gpu_module.body is region
    assert gpu_module.sym_name is name


def test_gpu_module_end():
    module_end = ModuleEndOp.get()

    assert isinstance(module_end, ModuleEndOp)
