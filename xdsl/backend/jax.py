from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jaxlib.mlir import ir

from xdsl.dialects.builtin import ModuleOp
from xdsl.traits import SymbolTable

P = ParamSpec("P")
R = TypeVar("R")


def jax_jit(module: ModuleOp) -> Callable[[Callable[P, R]], Callable[P, R]]:
    func_op = SymbolTable.lookup_symbol(module, "main")
    if func_op is None:
        raise ValueError("No `main` function in module.")

    program = str(module)

    mlir_module = ir.Module.parse(program, context=mlir.make_ir_context())
    bytecode = mlir.module_to_bytecode(mlir_module)
    client = xla_bridge.backends()["cpu"]
    loaded = client.compile(bytecode)

    def wrapper(stub: Callable[P, R]) -> Callable[P, R]:
        def single_element(*args: Any) -> R:
            return loaded.execute(args)[0]

        return single_element

    return wrapper
