from collections.abc import Callable, Sequence
from inspect import signature
from typing import Any, ParamSpec, cast, get_args, get_origin

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.typing import SupportsDType
from jaxlib.mlir import ir
from jaxlib.xla_client import LoadedExecutable
from typing_extensions import TypeVar

from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.traits import SymbolTable

P = ParamSpec("P")
R = TypeVar("R", bound=tuple[jnp.ndarray, ...] | jnp.ndarray)

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
    @JaxExecutable.compile(my_module)
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

    def __call__(self, stub: Callable[P, R]) -> Callable[P, R]:
        func_type = self.main_type
        loaded = self.loaded_executable

        operand_types = func_type.inputs.data
        result_types = func_type.outputs.data

        sig = signature(stub)

        if len(sig.parameters) != len(operand_types):
            raise ValueError(
                f"Number of parameters ({len(sig.parameters)}) does not match the number of operand types ({len(operand_types)})"
            )

        # Check that all parameters are annotated as jnp.ndarray
        for param in sig.parameters.values():
            if param.annotation != jnp.ndarray:
                raise NotImplementedError(
                    f"Parameter {param.name} is not annotated as jnp.ndarray"
                )

        # Check return annotation
        sig_return = sig.return_annotation
        sig_return_origin = get_origin(sig_return)

        if sig_return_origin is not tuple:
            if sig.return_annotation is not jnp.ndarray:
                raise NotImplementedError(
                    f"Return annotation is must be jnp.ndarray or a tuple of jnp.ndarray, got {sig_return}."
                )
            if len(result_types) != 1:
                raise ValueError(
                    f"Number of return values ({len(result_types)}) does not match the stub's return annotation"
                )

            def func(*args: P.args, **kwargs: P.kwargs) -> R:
                result = loaded.execute(args)
                return result[0]
        else:
            return_args = get_args(sig_return)
            if not all(return_arg is jnp.ndarray for return_arg in return_args):
                raise NotImplementedError(
                    f"Return annotation is must be jnp.ndarray or a tuple of jnp.ndarray, got {sig_return}."
                )

            def func(*args: P.args, **kwargs: P.kwargs) -> R:
                result = loaded.execute(args)
                return cast(R, tuple(result))

        return func
