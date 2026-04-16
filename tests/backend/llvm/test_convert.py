import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.test import TestOp

ir = pytest.importorskip("llvmlite.ir")
from xdsl.backend.llvm.convert import convert_module  # noqa: E402


def test_convert_empty_module():
    module = ModuleOp([])
    llvm_module = convert_module(module)
    assert isinstance(llvm_module, ir.Module)
    assert llvm_module.name == ""


def test_convert_module_with_op_raises():
    op = TestOp()
    module = ModuleOp([op])

    with pytest.raises(
        NotImplementedError, match="Conversion not implemented for op: test.op"
    ):
        convert_module(module)


def test_convert_module_target_triple():
    module = ModuleOp([])

    llvm_module = convert_module(module, target_triple="x86_64-unknown-linux-gnu")

    assert llvm_module.triple == "x86_64-unknown-linux-gnu"


def test_convert_module_data_layout():
    module = ModuleOp([])

    llvm_module = convert_module(
        module, data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64"
    )

    assert llvm_module.data_layout == "e-m:e-p270:32:32-p271:32:32-p272:64:64"


def test_convert_module_target_config_combined():
    module = ModuleOp([])

    llvm_module = convert_module(
        module,
        target_triple="x86_64-unknown-linux-gnu",
        data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64",
    )

    assert llvm_module.triple == "x86_64-unknown-linux-gnu"
    assert llvm_module.data_layout == "e-m:e-p270:32:32-p271:32:32-p272:64:64"
