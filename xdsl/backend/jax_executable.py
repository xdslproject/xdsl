from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.typing import SupportsDType
from jaxlib.mlir import ir
from jaxlib.xla_client import LoadedExecutable

from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.traits import SymbolTable

# JAX DTypeLike is currently broken
# The np.dtype annotation in jax does not specify the generic parameter
DTypeLike = (
    str  # like 'float32', 'int32'
    | type[Any]  # like np.float32, np.int32, float, int
    | np.dtype[Any]  # like np.dtype('float32'), np.dtype('int32')
    | SupportsDType  # like jnp.float32, jnp.int32
)


def array(object: Any, dtype: DTypeLike | None = None, copy: bool = True) -> Array:
    """
    Creates a jnp array with the passed-in parameters.

    JAX type annotations are currently broken, as they don't provide a generic parameter to `np.dtype`.
    This helper works around this issue.
    """
    return jnp.array(object, None, copy)  # pyright: ignore[reportUnknownMemberType]


class JaxExecutable:
    """
    A class wrapping a jax LoadedExecutable.

    Usage:

    ``` python
    my_module_str =\"""
    func.func @main(%arg0: tensor<5xf32>) -> tensor<5xf32> {
        ...
    }
    \"""
    my_module = parser.parse_module(my_module_str)
    @JaxExecutable.compile(module)
    def my_func(a: jax.Array) -> jax.Array: ...
    ```
    Note that the type of the function passed to the decorator is preserver, and can
    influence the behaviour of the function.
    Importantly, if the function has a single result, a one-element tuple or a single
    JAX array may be used as the return type annotation.
    The number of inputs and outputs of the `main` function in the module and the
    function being wrapped must match, or a `ValueError` is raised.
    """

    main_type: FunctionType
    loaded_executable: LoadedExecutable

    def __init__(
        self, main_type: FunctionType, loaded_executable: LoadedExecutable
    ) -> None:
        self.main_type = main_type
        self.loaded_executable = loaded_executable

    def execute(self, arguments: Sequence[Array]) -> list[Array]:
        return self.loaded_executable.execute(arguments)

    @staticmethod
    def compile(module: ModuleOp) -> "JaxExecutable":
        func_op = SymbolTable.lookup_symbol(module, "main")
        if func_op is None:
            raise ValueError("No `main` function in module.")
        if not isinstance(func_op, FuncOp):
            raise ValueError("`main` operation is not a `func.func`.")

        program = str(module)

        mlir_module = ir.Module.parse(program, context=mlir.make_ir_context())
        bytecode = mlir.module_to_bytecode(mlir_module)
        client = xla_bridge.backends()["cpu"]
        loaded = client.compile(bytecode)
        return JaxExecutable(func_op.function_type, loaded)
