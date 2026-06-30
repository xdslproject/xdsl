from dataclasses import dataclass
from typing import IO

import llvmlite.ir as ir

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.backend.llvm.convert_type import convert_type
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntAttr, IntegerAttr, ModuleOp, StringAttr
from xdsl.ir import Block, SSAValue
from xdsl.utils.exceptions import LLVMTranslationException
from xdsl.utils.target import Target

_ARG_ATTR_FLAGS = {
    "llvm.inreg": "inreg",
    "llvm.nest": "nest",
    "llvm.noalias": "noalias",
    "llvm.nocapture": "nocapture",
    "llvm.nofree": "nofree",
    "llvm.nonnull": "nonnull",
    "llvm.noundef": "noundef",
    "llvm.returned": "returned",
    "llvm.signext": "signext",
    "llvm.zeroext": "zeroext",
}

_ARG_ATTR_INTS = {
    "llvm.align": "align",
    "llvm.dereferenceable": "dereferenceable",
    "llvm.dereferenceable_or_null": "dereferenceable_or_null",
}

# Type-valued attrs need a typed pointer so llvmlite can read pointee.
_ARG_ATTR_TYPES = {
    "llvm.byval": "byval",
    "llvm.byref": "byref",
    "llvm.sret": "sret",
    "llvm.inalloca": "inalloca",
    "llvm.preallocated": "preallocated",
}


def _declare_func(op: llvm.FuncOp, llvm_module: ir.Module):
    ret_type = convert_type(op.function_type.output)
    arg_types: list[ir.Type] = []
    for idx, mlir_type in enumerate(op.function_type.inputs):
        if not (
            isinstance(mlir_type, llvm.LLVMPointerType) and op.arg_attrs is not None
        ):
            arg_types.append(convert_type(mlir_type))
            continue
        attrs = op.arg_attrs.data[idx].data
        elem = next((attrs[n] for n in _ARG_ATTR_TYPES if n in attrs), None)
        if elem is None:
            arg_types.append(convert_type(mlir_type))
            continue
        addrspace = (
            mlir_type.addr_space.data
            if isinstance(mlir_type.addr_space, IntAttr)
            else 0
        )
        arg_types.append(ir.PointerType(convert_type(elem), addrspace=addrspace))
    func_type = ir.FunctionType(
        ret_type, arg_types, var_arg=op.function_type.is_variadic
    )
    fn = ir.Function(llvm_module, func_type, name=op.sym_name.data)

    if op.arg_attrs is None:
        return
    for llvm_arg, attr_dict in zip(fn.args, op.arg_attrs):
        for mlir_name, value in attr_dict.data.items():
            if mlir_name in _ARG_ATTR_FLAGS:
                llvm_arg.add_attribute(_ARG_ATTR_FLAGS[mlir_name])
                continue
            if mlir_name in _ARG_ATTR_TYPES:
                llvm_arg.add_attribute(_ARG_ATTR_TYPES[mlir_name])
                continue
            if mlir_name not in _ARG_ATTR_INTS:
                continue
            assert isinstance(value, IntegerAttr)
            setattr(llvm_arg.attributes, _ARG_ATTR_INTS[mlir_name], value.value.data)


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
    *,
    fallback_target_triple: str | None,
    data_layout: str = "",
) -> ir.Module:
    """
    Convert an xDSL module to an LLVM module.

    Args:
        module: The xDSL module to convert.
        fallback_target_triple: The target triple to use when the module does not
            carry an ``llvm.target_triple`` attribute. The triple of the resulting
            module is resolved as follows:

            1. If the module has an ``llvm.target_triple`` attribute, its value is
               always used and ``fallback_target_triple`` is ignored.
            2. Otherwise, if ``fallback_target_triple`` is not ``None``, it is used
               as-is.
            3. Otherwise (it is ``None``), the host's default triple, as reported by
               ``llvmlite.binding.get_default_triple()``, is used.
        data_layout: The data layout to set on the resulting module. If empty, no
            data layout is set.

    Returns:
        The corresponding llvmlite IR module.

    Raises:
        LLVMTranslationException: If the ``llvm.target_triple`` attribute is present
            but is not a ``StringAttr``.
        NotImplementedError: If the module contains an op that is not a ``FuncOp``.
    """
    llvm_module = ir.Module()
    module_triple = module.attributes.get("llvm.target_triple")
    if module_triple is not None:
        if not isinstance(module_triple, StringAttr):
            raise LLVMTranslationException(
                f"Unsupported llvm.target_triple attribute: {module_triple}"
            )
        llvm_module.triple = module_triple.data
    else:
        if fallback_target_triple is None:
            from llvmlite import binding

            fallback_target_triple = binding.get_default_triple()
        llvm_module.triple = fallback_target_triple
    if data_layout:
        llvm_module.data_layout = data_layout

    func_ops: list[llvm.FuncOp] = []
    for op in module.ops:
        if not isinstance(op, llvm.FuncOp):
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
        func_ops.append(op)

    # Declare all functions (enables forward references)
    for op in func_ops:
        _declare_func(op, llvm_module)

    # Generate function bodies
    for func_op in func_ops:
        if func_op.body.blocks:
            _convert_func(func_op, llvm_module)

    return llvm_module


@dataclass(frozen=True)
class LLVMTarget(Target):
    name = "llvm"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        llvm_module = convert_module(module, fallback_target_triple=None)
        print(llvm_module, file=output)
