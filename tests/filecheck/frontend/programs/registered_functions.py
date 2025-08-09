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


@ctx.parse_program
def test_args():
    return add_i32(1, 2)  # pyright: ignore[reportArgumentType]


try:
    test_args.module
except CodeGenerationException as e:
    print(e.msg)
    # CHECK-NEXT: Function arguments must be declared variables.


# ================================================= #
# Disable the desymref pass for the remaining tests #
# ================================================= #
ctx.post_transforms = []


@ctx.parse_program
def test_args():
    return add_i32(operand1=1, operand2=2)  # pyright: ignore[reportArgumentType]


try:
    test_args.module
except CodeGenerationException as e:
    print(e.msg)
    # CHECK-NEXT: Function arguments must be declared variables.


def func():
    pass


@ctx.parse_program
def test_unregistered_func():
    return func()  # noqa: F821


try:
    test_unregistered_func.module
except CodeGenerationException as e:
    print(e.msg)
    # CHECK-NEXT: Function 'func' is not registered.


@ctx.parse_program
def test_missing_func():
    return func()  # noqa: F821


del func

try:
    test_missing_func.module
except CodeGenerationException as e:
    print(e.msg)
    # CHECK-NEXT: Function 'func' is not defined in scope.
