import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]

from xdsl.dialects.builtin import ModuleOp


def convert_module(module: ModuleOp) -> ir.Module:
    # TODO: map xDSL llvm dialect to llvmlite IR
    # see: https://llvmlite.readthedocs.io/en/latest/user-guide/ir/index.html
    # see: https://gist.github.com/sueszli/6aea9f54e7305e9e7dfdb31fd999b68e
    # demo: $ echo '"test.op"() : () -> ()' | uv run xdsl-opt -t llvm

    llvm_module = ir.Module()

    for op in module.ops:
        raise NotImplementedError(f"Conversion not implemented for op: {op.name}")

    return llvm_module
