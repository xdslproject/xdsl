import llvmlite.ir as ir

from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp


def _convert_func(op: llvm.FuncOp, llvm_module: ir.Module):
    ret_type = convert_type(op.function_type.output)
    arg_types = [convert_type(t) for t in op.function_type.inputs]
    func_type = ir.FunctionType(ret_type, arg_types)
    func_name = op.sym_name.data

    ir.Function(llvm_module, func_type, name=func_name)

    if op.body.blocks:
        raise NotImplementedError("Function definitions are not supported yet")


def convert_module(module: ModuleOp) -> ir.Module:
    """
    Convert an xDSL module to an LLVM module.
    """
    llvm_module = ir.Module()

    for op in module.ops:
        match op:
            case llvm.FuncOp():
                _convert_func(op, llvm_module)
            case _:
                raise NotImplementedError(
                    f"Conversion not implemented for op: {op.name}"
                )

    return llvm_module
