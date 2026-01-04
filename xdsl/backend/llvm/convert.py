import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]

from xdsl.dialects.builtin import ModuleOp


def convert_module(module: ModuleOp) -> ir.Module:
    """
    Convert an xDSL module to an LLVM module.
    """
    llvm_module = ir.Module()

    for op in module.ops:
        raise NotImplementedError(f"Conversion not implemented for op: {op.name}")

    return llvm_module
