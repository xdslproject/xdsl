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


class TypeMap(NamedTuple):
    """
    A helper containing the Python class, the corresponding ctypes class, and converter
    functions for instances of the Python class to ctypes and vice versa.
    """

    python_type: type[Any]
    ctype_type: type[Any]
    to_ctype: Callable[[Any], Any]
    from_ctype: Callable[[Any], Any]


class FuncTypeMap(NamedTuple):
    arg_maps: tuple[TypeMap, ...]
    res_map: TypeMap

    def c_func_type(self) -> "type[_CFunctionType]":
        return CFUNCTYPE(
            self.res_map.ctype_type, *(m.ctype_type for m in self.arg_maps)
        )


class CTypesTypeConverter:
    """
    Helper class to convert Python types and values to and from their c_types
    representation.
    Should be in sync with the conversion by the frontend.
    """

    _mapping: dict[type[Any], TypeMap]

    def __init__(self):
        self._mapping = {}

    def extend(self, type_map: TypeMap):
        self._mapping[type_map.python_type] = type_map

    def type_map(self, python_type: type[Any]) -> TypeMap:
        return self._mapping[python_type]

    def func_type_map(self, signature: TypeForm[Callable[P, R]]) -> FuncTypeMap:
        param_types, return_type = get_args(signature)
        return FuncTypeMap(
            tuple(self._mapping[py_type] for py_type in param_types),
            self._mapping[return_type],
        )


class CTypesAttributeConverter:
    """
    Helper class to convert Attributes in the IR to their c_types representation.
    The Python ->(frontend) IR ->(lowering) IR ->(this) ctypes conversion should be in
    sync with the CTypesTypeConverter used in the JIT.
    """

    _mapping: dict[type[ir.Attribute], Callable[[ir.Attribute], type[Any]]]

    def __init__(self):
        self._mapping = {}

    def extend(
        self,
        attribute_class: type[ir.Attribute],
        to_ctype: Callable[[ir.Attribute], type[Any]],
    ) -> None:
        self._mapping[attribute_class] = to_ctype

    def convert_type(self, attribute: ir.Attribute) -> type[Any]:
        return self._mapping[type(attribute)](attribute)

    def c_func_type_from_func_type(
        self, arg_types: tuple[ir.Attribute, ...], res_type: ir.Attribute
    ) -> "type[_CFunctionType]":
        return CFUNCTYPE(
            self.convert_type(res_type), *(self.convert_type(arg) for arg in arg_types)
        )


def wrap_jit_func(
    raw_func: RawJITFunc,
    original_func: Callable[P, R],
    signature: TypeForm[Callable[P, R]],
    c_types_type_converter: CTypesTypeConverter,
) -> WrappedJITFunc[P, R]:
    func_type_map = c_types_type_converter.func_type_map(signature)
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


# Backend


class JITBackend:
    def wrap(
        self,
        func: Callable[P, R],
        mlir_module: builtin.ModuleOp,
        symbol: str,
        c_types_type_converter: CTypesTypeConverter,
        c_types_attribute_converter: CTypesAttributeConverter,
        signature: TypeForm[Callable[P, R]],
    ) -> WrappedJITFunc[P, R]: ...


# Overall driver


class JITContext:
    pyast_ctx: PyASTContext
    c_types_type_converter: CTypesTypeConverter
    c_types_attribute_converter: CTypesAttributeConverter
    jit_backend: JITBackend

    def __init__(self, jit_backend: JITBackend):
        ctx = PyASTContext()
        self.pyast_ctx = ctx
        self.c_types_type_converter = CTypesTypeConverter()
        self.c_types_attribute_converter = CTypesAttributeConverter()
        self.jit_backend = jit_backend

    def jit(
        self, signature: TypeForm[Callable[P, R]]
    ) -> Callable[[Callable[P, R]], WrappedJITFunc[P, R]]:
        def inner(func: Callable[P, R]) -> WrappedJITFunc[P, R]:
            parsed_program = self.pyast_ctx.parse_program(func)
            return self.jit_backend.wrap(
                func,
                parsed_program.module,
                parsed_program.name,
                self.c_types_type_converter,
                self.c_types_attribute_converter,
                signature,
            )

        return inner


# LLVM-specific things


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


class LLVMJITBackend(JITBackend):
    def wrap(
        self,
        func: Callable[P, R],
        mlir_module: builtin.ModuleOp,
        symbol: str,
        c_types_type_converter: CTypesTypeConverter,
        c_types_attribute_converter: CTypesAttributeConverter,
        signature: TypeForm[Callable[P, R]],
    ) -> WrappedJITFunc[P, R]:
        func_op = SymbolTable.lookup_symbol(mlir_module, symbol)
        assert isinstance(func_op, llvm.FuncOp)
        xdsl_func_type = func_op.function_type
        c_func_type = c_types_attribute_converter.c_func_type_from_func_type(
            xdsl_func_type.inputs.data, xdsl_func_type.output
        )
        llvm_module = convert_module(mlir_module, fallback_target_triple=None)
        raw_func = llvm_jit(llvm_module, symbol, c_func_type)
        wrapped_func = wrap_jit_func(raw_func, func, signature, c_types_type_converter)
        return wrapped_func


# JIT

# TODO: add passes to xDSL
convert_to_llvm = MLIROptPass(
    arguments=("--convert-arith-to-llvm", "--convert-func-to-llvm"),
    generic=True,
)


# Test

ctx = JITContext(LLVMJITBackend())

# Register lowering to llvm
ctx.pyast_ctx.post_transforms = [FrontendDesymrefyPass(), convert_to_llvm]
ctx.pyast_ctx.register_type(float, builtin.f64)
ctx.pyast_ctx.register_function(float.__add__, arith.AddfOp)
ctx.pyast_ctx.register_dialect(arith.Arith)
ctx.pyast_ctx.register_dialect(llvm.LLVM)
ctx.pyast_ctx.register_dialect(builtin.Builtin)
ctx.pyast_ctx.register_dialect(func.Func)

# Register Python and IR type conversion
ctx.c_types_type_converter.extend(TypeMap(float, c_double, c_double, float))
ctx.c_types_attribute_converter.extend(builtin.Float64Type, lambda _: c_double)

# Use


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
