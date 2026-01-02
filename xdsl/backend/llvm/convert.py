import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]

from xdsl.dialects.builtin import Float64Type, IntegerType, ModuleOp
from xdsl.dialects.llvm import FAddOp, FuncOp, LLVMVoidType, ReturnOp
from xdsl.ir import Attribute, Operation, SSAValue


class LLVMTranslationException(Exception):
    pass


def convert_type(type_attr: Attribute) -> ir.Type:
    match type_attr:
        case IntegerType():
            return ir.IntType(type_attr.bitwidth)  # pyright: ignore[reportUnknownVariableType]
        case Float64Type():
            return ir.DoubleType()
        case LLVMVoidType():
            return ir.VoidType()
        case _:
            raise LLVMTranslationException(f"Type not supported: {type_attr}")


def _convert_op(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    match op:
        case FAddOp():
            lhs = val_map[op.lhs]
            rhs = val_map[op.rhs]
            res = builder.fadd(lhs, rhs, name="res")  # pyright: ignore[reportUnknownMemberType]
            val_map[op.res] = res  # pyright: ignore[reportArgumentType]
        case ReturnOp():
            if op.operands:
                builder.ret(val_map[op.operands[0]])  # pyright: ignore[reportUnknownMemberType]
            else:
                builder.ret_void()
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")


def _convert_func(func_op: FuncOp, llvm_module: ir.Module):
    func_type_attr = func_op.function_type
    ret_type = convert_type(func_type_attr.output)
    arg_types = [convert_type(t) for t in func_type_attr.inputs]
    func_type = ir.FunctionType(ret_type, arg_types)
    func = ir.Function(llvm_module, func_type, name=func_op.sym_name.data)

    val_map: dict[SSAValue, ir.Value] = {}

    for i, block in enumerate(func_op.body.blocks):
        llvm_block = func.append_basic_block(name=f"block_{i}")
        builder_block = ir.IRBuilder(llvm_block)

        is_entry = i == 0
        if is_entry:
            for arg, llvm_arg in zip(block.args, func.args):
                val_map[arg] = llvm_arg
        else:
            # for now we assume simple CFG
            pass

        for op in block.ops:
            _convert_op(op, builder_block, val_map)


def convert_module(module: ModuleOp) -> ir.Module:
    """
    Convert an xDSL module to an LLVM module.
    """
    llvm_module = ir.Module()

    for op in module.ops:
        match op:
            case FuncOp():
                _convert_func(op, llvm_module)
            case _:
                raise NotImplementedError(
                    f"Conversion not implemented for op: {op.name}"
                )

    return llvm_module
