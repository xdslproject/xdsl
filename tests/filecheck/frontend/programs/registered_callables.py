# RUN: python %s | filecheck %s

from ctypes import c_int32

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException


def add_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...


class Adder:
    @classmethod
    def add_i32(cls, operand1: c_int32, operand2: c_int32) -> c_int32: ...


ctx = PyASTContext()
ctx.register_type(c_int32, builtin.i32)
ctx.register_function(add_i32, arith.AddiOp)
ctx.register_function(Adder.add_i32, arith.AddiOp)


@ctx.parse_program
def test_add_function(x: c_int32, y: c_int32) -> c_int32:
    return add_i32(x, operand2=y)


print(test_add_function.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_add_function(%x : i32, %y : i32) -> i32 {
# CHECK-NEXT:     %0 = arith.addi %x, %y : i32
# CHECK-NEXT:     func.return %0 : i32
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_add_classmethod(x: c_int32, y: c_int32) -> c_int32:
    return Adder.add_i32(x, operand2=y)


print(test_add_classmethod.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_add_classmethod(%x : i32, %y : i32) -> i32 {
# CHECK-NEXT:     %0 = arith.addi %x, %y : i32
# CHECK-NEXT:     func.return %0 : i32
# CHECK-NEXT:   }
# CHECK-NEXT: }


# CHECK-NEXT: Callable arguments must be declared variables.
@ctx.parse_program
def test_args_function():
    return add_i32(1, 2)  # pyright: ignore[reportArgumentType]


try:
    test_args_function.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK-NEXT: Callable arguments must be declared variables.
@ctx.parse_program
def test_args_classmethod():
    return Adder.add_i32(1, 2)  # pyright: ignore[reportArgumentType]


try:
    test_args_classmethod.module
except CodeGenerationException as e:
    print(e.msg)


# ================================================= #
# Disable the desymref pass for the remaining tests #
# ================================================= #
ctx.post_transforms = []


# CHECK-NEXT: Callable arguments must be declared variables.
@ctx.parse_program
def test_more_args_function():
    return add_i32(operand1=1, operand2=2)  # pyright: ignore[reportArgumentType]


try:
    test_more_args_function.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK-NEXT: Callable arguments must be declared variables.
@ctx.parse_program
def test_more_args_classmethod():
    return Adder.add_i32(operand1=1, operand2=2)  # pyright: ignore[reportArgumentType]


try:
    test_more_args_classmethod.module
except CodeGenerationException as e:
    print(e.msg)


def func():
    pass


class Class:
    @classmethod
    def method(cls):
        pass


# CHECK-NEXT: Callable 'func' is not registered.
@ctx.parse_program
def test_unregistered_func():
    return func()  # noqa: F821


try:
    test_unregistered_func.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK-NEXT: Callable 'Class.method' is not registered.
@ctx.parse_program
def test_unregistered_classmethod():
    return Class.method()  # noqa: F821


try:
    test_unregistered_classmethod.module
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


# CHECK-NEXT: Method 'method' is not defined on class 'Class'.
@ctx.parse_program
def test_missing_method():
    return Class.method()  # noqa: F821


del Class.method

try:
    test_missing_method.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK-NEXT: Class 'Class' is not defined in scope.
@ctx.parse_program
def test_missing_class():
    return Class.method()  # noqa: F821


del Class

try:
    test_missing_class.module
except CodeGenerationException as e:
    print(e.msg)
