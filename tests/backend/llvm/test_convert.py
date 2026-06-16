import pytest

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region

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


def test_convert_module_declaration():
    # a func op with no body becomes a declaration
    ft = llvm.LLVMFunctionType([i32], i32)
    func = llvm.FuncOp("my_decl", ft)
    module = ModuleOp([func])

    llvm_module = convert_module(module)
    fn = llvm_module.get_global("my_decl")
    assert fn is not None
    assert not fn.basic_blocks


def test_convert_module_forward_reference():
    # a function can call another function defined later in the module
    ft_callee = llvm.LLVMFunctionType([i32], i32)
    ft_caller = llvm.LLVMFunctionType([i32], i32)

    caller_block = Block(arg_types=[i32])
    arg = caller_block.args[0]
    call_op = llvm.CallOp("callee", arg, return_type=i32)
    caller_block.add_op(call_op)
    ret_op = llvm.ReturnOp(call_op.returned)
    caller_block.add_op(ret_op)
    caller_body = Region(caller_block)

    caller = llvm.FuncOp("caller", ft_caller, body=caller_body)

    callee_block = Block(arg_types=[i32])
    callee_ret = llvm.ReturnOp(callee_block.args[0])
    callee_block.add_op(callee_ret)
    callee_body = Region(callee_block)

    callee = llvm.FuncOp("callee", ft_callee, body=callee_body)

    # caller defined before callee
    module = ModuleOp([caller, callee])
    llvm_module = convert_module(module)

    caller_fn = llvm_module.get_global("caller")
    callee_fn = llvm_module.get_global("callee")
    assert caller_fn is not None
    assert callee_fn is not None
    assert caller_fn.basic_blocks
    assert callee_fn.basic_blocks
