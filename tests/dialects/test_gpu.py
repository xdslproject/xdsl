from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.dialects.gpu import AllReduceOperationAttr, ModuleEndOp, ModuleOp, DimensionAttr


def test_dimension():
    dim = DimensionAttr.from_dimension("x")

    assert dim.value.param.data == "x"


def test_all_reduce_operation():
    op = AllReduceOperationAttr.from_op("add")

    assert op.value.param.data == "add"


def test_gpu_module():
    name = SymbolRefAttr.from_str("gpu")

    ops = [ModuleEndOp.get()]

    gpu_module = ModuleOp.get(name, ops)

    assert gpu_module.body.ops == ops
    assert gpu_module.sym_name is name


def test_gpu_module_end():
    module_end = ModuleEndOp.get()

    assert isinstance(module_end, ModuleEndOp)
