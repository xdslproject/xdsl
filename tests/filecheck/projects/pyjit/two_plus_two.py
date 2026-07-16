# RUN: python %s | filecheck %s

from collections.abc import Callable
from ctypes import CFUNCTYPE, c_double
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, ParamSpec, get_args

import llvmlite
import llvmlite.binding
import llvmlite.ir as llvm_ir
from typing_extensions import TypeForm, TypeVar

from xdsl import ir
from xdsl.backend.llvm.convert import convert_module
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.traits import SymbolTable
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.transforms.mlir_opt import MLIROptPass

if TYPE_CHECKING:
    from ctypes import _CFunctionType  # pyright: ignore[reportPrivateUsage]

# Lowering

# TODO: add passes to xDSL
convert_to_llvm = MLIROptPass(
    arguments=("--convert-arith-to-llvm", "--convert-func-to-llvm"),
    generic=True,
)

# Executable


@dataclass(slots=True)
class RawJITFunc:
    c_func_type: "type[_CFunctionType]"
    c_func: "_CFunctionType"


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class WrappedJITFunc(Generic[P, R]):
    raw_func: RawJITFunc
    original_func: Callable[P, R]
    __call__: Callable[P, R]


@dataclass(slots=True, init=False)
class LLVMRawJITFunc(RawJITFunc):
    """Holds LLVM MCJIT-owned objects so jitted code is not unmapped by GC."""

    target: object
    target_machine: object
    backing_mod: object
    engine: object

    def __init__(
        self,
        c_func_type: "type[_CFunctionType]",
        c_func: "_CFunctionType",
        target: object,
        target_machine: object,
        backing_mod: object,
        engine: object,
    ):
        super(LLVMRawJITFunc, self).__init__(c_func_type, c_func)
        self.target = target
        self.target_machine = target_machine
        self.backing_mod = backing_mod
        self.engine = engine


class TypeMap(NamedTuple):
    python_type: type[Any]
    ctype_type: type[Any]
    to_ctype: Callable[[Any], Any]
    from_ctype: Callable[[Any], Any]


bla: dict[type[Any], TypeMap] = {float: TypeMap(float, c_double, c_double, float)}


class FuncTypeMap(NamedTuple):
    arg_maps: tuple[TypeMap, ...]
    res_map: TypeMap

    @staticmethod
    def from_signature(signature: TypeForm[Callable[P, R]]) -> "FuncTypeMap":
        param_types, return_type = get_args(signature)
        return FuncTypeMap(
            tuple(bla[py_type] for py_type in param_types), bla[return_type]
        )

    def c_func_type(self):
        return CFUNCTYPE(
            self.res_map.ctype_type, *(m.ctype_type for m in self.arg_maps)
        )


class CTypeConverter:
    def convert_type(self, attribute: ir.Attribute) -> Any:
        assert attribute == builtin.f64
        return c_double

    def c_func_type_from_func_type(
        self, arg_types: tuple[ir.Attribute, ...], res_type: ir.Attribute
    ) -> "type[_CFunctionType]":
        return CFUNCTYPE(
            self.convert_type(res_type), *(self.convert_type(arg) for arg in arg_types)
        )


def wrapped(
    raw_func: RawJITFunc,
    original_func: Callable[P, R],
    signature: TypeForm[Callable[P, R]],
) -> WrappedJITFunc[P, R]:
    func_type_map = FuncTypeMap.from_signature(signature)
    assert raw_func.c_func_type == func_type_map.c_func_type(), (
        f"CTypes signature inferred from frontend ({raw_func.c_func_type}) does not "
        f"match signature from JIT ({func_type_map.c_func_type()})."
    )

    def fn(*args: P.args, **kwargs: P.kwargs) -> R:
        assert not kwargs
        ctype_args = tuple(
            m.to_ctype(a) for m, a in zip(func_type_map.arg_maps, args, strict=True)
        )
        ctype_res = raw_func.c_func(*ctype_args)
        return func_type_map.res_map.from_ctype(ctype_res)

    return WrappedJITFunc(raw_func, original_func, fn)


def llvm_jit(
    llvm_module: llvm_ir.Module, symbol: str, c_func_type: "type[_CFunctionType]"
) -> LLVMRawJITFunc:
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
    c_types_fn = c_func_type(func_ptr)  # pyright: ignore

    keepalive = LLVMRawJITFunc(
        c_func_type,
        c_types_fn,
        target=target,  # pyright: ignore
        target_machine=target_machine,  # pyright: ignore
        backing_mod=backing_mod,  # pyright: ignore
        engine=engine,  # pyright: ignore
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
    ) -> Callable[[Callable[P, R]], WrappedJITFunc[P, R]]:
        def inner(func: Callable[P, R]) -> WrappedJITFunc[P, R]:
            parsed_program = self.pyast_ctx.parse_program(func)
            mlir_module = parsed_program.module
            func_op = SymbolTable.lookup_symbol(mlir_module, parsed_program.name)
            assert isinstance(func_op, llvm.FuncOp)
            xdsl_func_type = func_op.function_type
            c_func_type = CTypeConverter().c_func_type_from_func_type(
                xdsl_func_type.inputs.data, xdsl_func_type.output
            )
            llvm_module = convert_module(mlir_module, fallback_target_triple=None)
            raw_func = llvm_jit(llvm_module, parsed_program.name, c_func_type)
            wrapped_func = wrapped(raw_func, func, signature)
            return wrapped_func

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

# CHECK: plus.original_func(2.0, 2.0) = 4.0
# CHECK: plus.original_func(3.0, 4.0) = 7.0
print(f"{plus.original_func(2.0, 2.0) = }")
print(f"{plus.original_func(3.0, 4.0) = }")

# CHECK: plus.raw_func.c_func(2.0, 2.0) = 4.0
# CHECK: plus.raw_func.c_func(3.0, 4.0) = 7.0
print(f"{plus.raw_func.c_func(2.0, 2.0) = }")
print(f"{plus.raw_func.c_func(3.0, 4.0) = }")
