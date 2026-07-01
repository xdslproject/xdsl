# RUN: python %s | filecheck %s

from collections.abc import Callable
from ctypes import CFUNCTYPE, c_double
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, ParamSpec, get_args

import llvmlite
import llvmlite.binding
import llvmlite.ir as llvm_ir
from typing_extensions import TypeForm, TypeVar

from xdsl.backend.llvm.convert import convert_module
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.transforms.mlir_opt import MLIROptPass

# Lowering

# TODO: add passes to xDSL
convert_to_llvm = MLIROptPass(
    arguments=("--convert-arith-to-llvm", "--convert-func-to-llvm"),
    generic=True,
)

# Executable

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class McJitKeepalive(Generic[P, R]):
    """Holds LLVM MCJIT-owned objects so jitted code is not unmapped by GC."""

    target: object
    target_machine: object
    backing_mod: object
    engine: object
    func: Callable[P, R]
    c_types_func: Callable[..., object]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.func(*args, **kwargs)


class TypeMap(NamedTuple):
    python_type: type[Any]
    ctype_type: type[Any]
    to_ctype: Callable[[Any], Any]
    from_ctype: Callable[[Any], Any]


bla: dict[type[Any], TypeMap] = {float: TypeMap(float, c_double, c_double, float)}


# TODO: support automatic conversion of types
def mcjit_binary(
    llvm_module: llvm_ir.Module, symbol: str, t: TypeForm[Callable[P, R]]
) -> McJitKeepalive[P, R]:
    llvm_ir_text = str(llvm_module)
    llvmlite.binding.initialize_native_target()  # pyright: ignore
    llvmlite.binding.initialize_native_asmprinter()  # pyright: ignore

    target = llvmlite.binding.Target.from_default_triple()  # pyright: ignore
    target_machine = target.create_target_machine()  # pyright: ignore
    backing_mod = llvmlite.binding.parse_assembly(llvm_ir_text)  # pyright: ignore
    engine = llvmlite.binding.create_mcjit_compiler(backing_mod, target_machine)  # pyright: ignore
    engine.finalize_object()  # pyright: ignore
    engine.run_static_constructors()  # pyright: ignore

    func_ptr = engine.get_function_address(symbol)  # pyright: ignore

    # Create mapping
    param_types, return_type = get_args(t)
    param_maps = tuple(bla[py_type] for py_type in param_types)
    return_map = bla[return_type]

    fn_type = CFUNCTYPE(return_map.ctype_type, *(m.ctype_type for m in param_maps))
    c_types_fn = fn_type(func_ptr)  # pyright: ignore

    def fn(*args: P.args, **kwargs: P.kwargs) -> R:
        ctype_args = tuple(m.to_ctype(a) for m, a in zip(param_maps, args, strict=True))
        return return_map.from_ctype(c_types_fn(*ctype_args))

    keepalive = McJitKeepalive[P, R](
        target=target,  # pyright: ignore
        target_machine=target_machine,  # pyright: ignore
        backing_mod=backing_mod,  # pyright: ignore
        engine=engine,  # pyright: ignore
        func=fn,
        c_types_func=c_types_fn,
    )

    return keepalive


# JIT


# TODO: support extending the JIT with more functionality
class JITContext:
    pyast_ctx: PyASTContext

    def __init__(self):
        ctx = PyASTContext(post_transforms=[FrontendDesymrefyPass(), convert_to_llvm])
        ctx.register_type(float, builtin.f64)
        ctx.register_function(float.__add__, arith.AddfOp)
        ctx.register_dialect(arith.Arith)
        ctx.register_dialect(llvm.LLVM)
        ctx.register_dialect(builtin.Builtin)
        ctx.register_dialect(func.Func)
        self.pyast_ctx = ctx

    def jit(
        self, signature: TypeForm[Callable[P, R]]
    ) -> Callable[[Callable[P, R]], McJitKeepalive[P, R]]:
        def inner(func: Callable[P, R]) -> McJitKeepalive[P, R]:
            parsed_program = self.pyast_ctx.parse_program(func)
            module = convert_module(parsed_program.module, fallback_target_triple=None)
            return mcjit_binary(module, parsed_program.name, signature)

        return inner


# Test

ctx = JITContext()


@ctx.jit(Callable[[float, float], float])
def plus(a: float, b: float) -> float:
    return a + b


# CHECK: plus(2.0, 2.0) = 4.0
# CHECK: plus(3.0, 4.0) = 7.0
print(f"{plus(2.0, 2.0) = }")
print(f"{plus(3.0, 4.0) = }")

# CHECK: plus.func(2.0, 2.0) = 4.0
# CHECK: plus.func(3.0, 4.0) = 7.0
print(f"{plus.func(2.0, 2.0) = }")
print(f"{plus.func(3.0, 4.0) = }")

# CHECK: plus.func(2.0, 2.0) = 4.0
# CHECK: plus.func(3.0, 4.0) = 7.0
print(f"{plus.c_types_func(2.0, 2.0) = }")
print(f"{plus.c_types_func(3.0, 4.0) = }")
