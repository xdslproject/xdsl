from dataclasses import dataclass
from typing import IO

import llvmlite.ir as ir

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.backend.llvm.convert_type import convert_type
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, SSAValue
from xdsl.utils.target import Target


def _convert_func(op: llvm.FuncOp, llvm_module: ir.Module):
    func = llvm_module.get_global(op.sym_name.data)

    if not op.body.blocks:
        return

    block_map: dict[Block, ir.Block] = {}
    val_map: dict[SSAValue, ir.Value] = {}

    # Create all blocks first so that forward references work
    for i, block in enumerate(op.body.blocks):
        llvm_block = func.append_basic_block(name=block.name_hint or "")
        block_map[block] = llvm_block
        if i == 0:
            for arg, llvm_arg in zip(block.args, func.args):
                val_map[arg] = llvm_arg

    # Create PHI nodes for non-entry block arguments
    # Incoming values are added later by branch ops (e.g. BrOp, CondBrOp) in convert_op
    for i, block in enumerate(op.body.blocks):
        if i == 0:
            continue
        if block.args:
            builder = ir.IRBuilder(block_map[block])
            for arg in block.args:
                phi = builder.phi(convert_type(arg.type))
                val_map[arg] = phi

    # Convert ops in each block
    for block in op.body.blocks:
        builder = ir.IRBuilder(block_map[block])
        # Position after any PHI nodes
        if block_map[block].instructions:
            builder.position_after(block_map[block].instructions[-1])
        for op_in_block in block.ops:
            convert_op(op_in_block, builder, val_map, block_map)


def convert_module(
    module: ModuleOp,
    target_triple: str = "",
    data_layout: str = "",
) -> ir.Module:
    """
    Convert an xDSL module to an LLVM module.
    """
    llvm_module = ir.Module()
    if target_triple:
        llvm_module.triple = target_triple
    if data_layout:
        llvm_module.data_layout = data_layout

    func_ops: list[llvm.FuncOp] = []
    for op in module.ops:
        if not isinstance(op, llvm.FuncOp):
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
        func_ops.append(op)

    # Declare all functions (enables forward references)
    for op in func_ops:
        ret_type = convert_type(op.function_type.output)
        arg_types = [convert_type(t) for t in op.function_type.inputs]
        func_type = ir.FunctionType(ret_type, arg_types)
        ir.Function(llvm_module, func_type, name=op.sym_name.data)

    # Generate function bodies
    for func_op in func_ops:
        if func_op.body.blocks:
            _convert_func(func_op, llvm_module)

    return llvm_module


@dataclass(frozen=True)
class LLVMTarget(Target):
    name = "llvm"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        llvm_module = convert_module(module)
        print(llvm_module, file=output)
