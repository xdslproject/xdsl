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

    block_map: dict[Block, ir.Block] = {}
    val_map: dict[SSAValue, ir.Value] = {}

    # create all blocks first so that forward references work
    for i, block in enumerate(op.body.blocks):
        llvm_block = func.append_basic_block(name=block.name_hint or "")
        block_map[block] = llvm_block
        if i == 0:
            for arg, llvm_arg in zip(block.args, func.args):
                val_map[arg] = llvm_arg

    # create PHI nodes for non-entry block arguments
    # incoming values are added later by branch ops (e.g. CondBrOp) in convert_op
    for i, block in enumerate(op.body.blocks):
        if i == 0:
            continue
        if block.args:
            builder = ir.IRBuilder(block_map[block])
            for arg in block.args:
                phi = builder.phi(convert_type(arg.type))
                val_map[arg] = phi

    # convert ops in each block
    for block in op.body.blocks:
        builder = ir.IRBuilder(block_map[block])
        # position after any PHI nodes
        if block_map[block].instructions:
            builder.position_after(block_map[block].instructions[-1])
        for op_in_block in block.ops:
            convert_op(op_in_block, builder, val_map, block_map)


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
