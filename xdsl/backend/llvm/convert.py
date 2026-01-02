from functools import cache

import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]

from xdsl.dialects.builtin import Float64Type, IntegerType, ModuleOp
from xdsl.dialects.llvm import GlobalOp, LLVMVoidType
from xdsl.ir import Attribute


class LLVMTranslationException(Exception):
    pass


@cache
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


def convert_module(module: ModuleOp) -> ir.Module:
    """
    Convert an xDSL module to an LLVM module.
    """
    llvm_module = ir.Module()

    for op in module.ops:
        match op:
            case GlobalOp():
                ir.GlobalVariable(
                    llvm_module, convert_type(op.global_type), name=op.sym_name.data
                )
            case _:
                raise NotImplementedError(
                    f"Conversion not implemented for op: {op.name}"
                )

    return llvm_module
