import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]

from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.llvm import GlobalOp


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
