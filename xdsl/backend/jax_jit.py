from collections.abc import Callable
from inspect import signature
from typing import ParamSpec, TypeVar, cast, get_args, get_origin

import jax.numpy as jnp
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jaxlib.mlir import ir

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.traits import SymbolTable

P = ParamSpec("P")
R = TypeVar("R", bound=tuple[jnp.ndarray, ...] | jnp.ndarray)


def jax_jit(module: ModuleOp) -> Callable[[Callable[P, R]], Callable[P, R]]:
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

    func_type = func_op.function_type
    operand_types = func_type.inputs.data
    result_types = func_type.outputs.data

    def wrapper(stub: Callable[P, R]) -> Callable[P, R]:
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

        if sig_return_origin is tuple:
            return_args = get_args(sig_return)
            if len(return_args) != 1:
                raise NotImplementedError("Only return values of length 1 supported")
            if return_args[0] is not jnp.ndarray:
                raise NotImplementedError(
                    f"Return annotation {return_args[0]} is not jnp.ndarray"
                )

            def func(*args: P.args, **kwargs: P.kwargs) -> R:
                result = loaded.execute(args)
                return cast(R, tuple(result))
        else:
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

        return func

    return wrapper
