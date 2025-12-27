import pytest

from xdsl.backend.llvmlite.convert import convert_module
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.test import TestOp

ir = pytest.importorskip("llvmlite.ir")


def test_convert_empty_module():
    module = ModuleOp([])
    llvm_module = convert_module(module)
    assert isinstance(llvm_module, ir.Module)
    assert isinstance(str(llvm_module), str)
    assert '; ModuleID = ""' in str(llvm_module)


def test_convert_module_with_op_raises():
    op = TestOp()
    module = ModuleOp([op])

    with pytest.raises(
        NotImplementedError, match="Conversion not implemented for op: test.op"
    ):
        convert_module(module)
