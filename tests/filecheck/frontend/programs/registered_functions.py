# RUN: python %s | filecheck %s

from ctypes import c_int32

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException


def add_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...


ctx = PyASTContext()
ctx.register_type(c_int32, builtin.i32)
ctx.register_function(add_i32, arith.AddiOp)


@ctx.parse_program
def test_add(x: c_int32, y: c_int32) -> c_int32:
    return add_i32(x, operand2=y)


print(test_add.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_add(%x : i32, %y : i32) -> i32 {
# CHECK-NEXT:     %0 = arith.addi %x, %y : i32
# CHECK-NEXT:     func.return %0 : i32
# CHECK-NEXT:   }
# CHECK-NEXT: }


# Test with float literal arguments
def add_f64(a: float, b: float) -> float: ...
ctx.register_type(float, builtin.f64)
ctx.register_function(add_f64, arith.AddfOp)
# Register float operations for expressions
ctx.register_function(float.__add__, arith.AddfOp)
ctx.register_function(float.__mul__, arith.MulfOp)

@ctx.parse_program
def test_float_literals() -> float:
    return add_f64(1.5, 2.5)


print(test_float_literals.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_float_literals() -> f64 {
# CHECK-NEXT:     %0 = arith.constant 1.500000e+00 : f64
# CHECK-NEXT:     %1 = arith.constant 2.500000e+00 : f64
# CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
# CHECK-NEXT:     func.return %2 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }


# Test with bool literal arguments  
def and_i1(a: bool, b: bool) -> bool: ...
ctx.register_type(bool, builtin.i1)
ctx.register_function(and_i1, arith.AndIOp)

@ctx.parse_program
def test_bool_literals() -> bool:
    return and_i1(True, False)


print(test_bool_literals.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_bool_literals() -> i1 {
# CHECK-NEXT:     %0 = arith.constant 1 : i1
# CHECK-NEXT:     %1 = arith.constant 0 : i1
# CHECK-NEXT:     %2 = arith.andi %0, %1 : i1
# CHECK-NEXT:     func.return %2 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }


# Test with expressions as arguments (mixing literals and variables)
@ctx.parse_program
def test_mixed_expr(x: float, y: float) -> float:
    return add_f64(x + 1.0, y * 2.0)


print(test_mixed_expr.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_mixed_expr(%x : f64, %y : f64) -> f64 {
# CHECK-NEXT:     %0 = arith.constant 1.000000e+00 : f64
# CHECK-NEXT:     %1 = arith.addf %x, %0 : f64
# CHECK-NEXT:     %2 = arith.constant 2.000000e+00 : f64
# CHECK-NEXT:     %3 = arith.mulf %y, %2 : f64
# CHECK-NEXT:     %4 = arith.addf %1, %3 : f64
# CHECK-NEXT:     func.return %4 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }


# Test with nested function calls as arguments
@ctx.parse_program
def test_nested_calls(x: float, y: float) -> float:
    return add_f64(add_f64(x, y), add_f64(x, y))


print(test_nested_calls.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_nested_calls(%x : f64, %y : f64) -> f64 {
# CHECK-NEXT:     %0 = arith.addf %x, %y : f64
# CHECK-NEXT:     %1 = arith.addf %x, %y : f64
# CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
# CHECK-NEXT:     func.return %2 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }


# ================================================= #
# Disable the desymref pass for the remaining tests #
# ================================================= #
ctx.post_transforms = []


# Test keyword arguments with float literals - now should work!
@ctx.parse_program
def test_kwargs_float_literals() -> float:
    x = 10.0
    y = 20.0
    return add_f64(x, y)


print(test_kwargs_float_literals.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_kwargs_float_literals() -> f64 {
# CHECK-NEXT:     "symref.declare"() {"symbol" = "x"} : () -> ()
# CHECK-NEXT:     %0 = arith.constant 1.000000e+01 : f64
# CHECK-NEXT:     "symref.update"(%0) {"symbol" = "x"} : (f64) -> ()
# CHECK-NEXT:     "symref.declare"() {"symbol" = "y"} : () -> ()
# CHECK-NEXT:     %1 = arith.constant 2.000000e+01 : f64
# CHECK-NEXT:     "symref.update"(%1) {"symbol" = "y"} : (f64) -> ()
# CHECK-NEXT:     %x = "symref.fetch"() {"symbol" = "x"} : () -> f64
# CHECK-NEXT:     %y = "symref.fetch"() {"symbol" = "y"} : () -> f64
# CHECK-NEXT:     %2 = arith.addf %x, %y : f64
# CHECK-NEXT:     func.return %2 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }


def func():
    pass


# CHECK-NEXT: Function 'func' is not registered.
@ctx.parse_program
def test_unregistered_func():
    return func()  # noqa: F821


try:
    test_unregistered_func.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK-NEXT: Function 'func' is not defined in scope.
@ctx.parse_program
def test_missing_func():
    return func()  # noqa: F821


del func

try:
    test_missing_func.module
except CodeGenerationException as e:
    print(e.msg)
