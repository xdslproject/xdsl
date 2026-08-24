import ctypes
from collections.abc import Callable

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.frontend.pyast.program import PyASTProgram
from xdsl.jit.context import JITBackend, JITContext, register_builtin_type_maps
from xdsl.jit.function import RawJITFunc
from xdsl.jit.py_type_context import TypeMap
from xdsl.traits import SymbolTable


def subtract(lhs: float, rhs: float, /) -> float:
    return lhs - rhs


class StubJITBackend(JITBackend):
    # records its inputs and binds `subtract` instead of compiling
    mlir_module: builtin.ModuleOp
    symbol: str
    ir_context: Context
    raw_func: RawJITFunc

    def jit(
        self,
        mlir_module: builtin.ModuleOp,
        symbol: str,
        ir_context: Context,
    ) -> RawJITFunc:
        self.mlir_module = mlir_module
        self.symbol = symbol
        self.ir_context = ir_context
        c_func_type = ctypes.CFUNCTYPE(
            ctypes.c_double, ctypes.c_double, ctypes.c_double
        )
        self.raw_func = RawJITFunc(c_func_type, c_func_type(subtract))
        return self.raw_func


@pytest.fixture
def backend() -> StubJITBackend:
    return StubJITBackend()


@pytest.fixture
def jit_context(backend: StubJITBackend) -> JITContext:
    context = JITContext(backend)
    context.pyast_ctx.register_function(float.__add__, arith.AddfOp)
    register_builtin_type_maps(context)
    return context


def test_jit_compiles_and_wraps(jit_context: JITContext, backend: StubJITBackend):
    @jit_context.jit(Callable[[float, float], float])
    def plus(a: float, b: float) -> float:
        return a + b

    assert backend.symbol == "plus"
    assert backend.ir_context is jit_context.pyast_ctx.ir_context
    assert isinstance(
        SymbolTable.lookup_symbol(backend.mlir_module, "plus"), func.FuncOp
    )
    assert plus(3.0, 4.0) == -1.0  # `subtract` and not `plus` in mock backend
    assert plus.raw_func is backend.raw_func
    assert not isinstance(plus.original_func, PyASTProgram)


def test_each_backend_owns_its_c_type_context():
    first, second = StubJITBackend(), StubJITBackend()
    first.c_type_context.register_ctype(builtin.Float64Type, lambda _: ctypes.c_double)
    assert second.c_type_context.registry == {}


def test_register_builtin_type_maps(jit_context: JITContext):
    assert jit_context.pyast_ctx.type_registry.get_annotation(builtin.f64) is float
    assert jit_context.py_type_context.type_map(float) == TypeMap(
        float, ctypes.c_double, ctypes.c_double, float
    )
