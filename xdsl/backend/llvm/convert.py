import llvmlite.ir as ir

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, SSAValue


def _convert_func(op: llvm.FuncOp, llvm_module: ir.Module):
    ret_type = convert_type(op.function_type.output)
    arg_types = [convert_type(t) for t in op.function_type.inputs]
    func_type = ir.FunctionType(ret_type, arg_types)
    func_name = op.sym_name.data

    func = ir.Function(llvm_module, func_type, name=func_name)

    if not op.body.blocks:
        return

    if len(op.body.blocks) > 1:
        raise NotImplementedError("Only single-block functions are supported")

    block_map: dict[Block, ir.Block] = {}
    val_map: dict[SSAValue, ir.Value] = {}

    entry_block = op.body.blocks[0]

    # entry block
    llvm_entry = func.append_basic_block(name=entry_block.name_hint or "")
    block_map[entry_block] = llvm_entry
    for arg, llvm_arg in zip(entry_block.args, func.args):
        val_map[arg] = llvm_arg

    # convert ops
    builder = ir.IRBuilder(llvm_entry)
    for op_in_block in entry_block.ops:
        convert_op(op_in_block, builder, val_map)


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
