# RUN: python %s | filecheck %s

from ctypes import c_int32

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException


class Adder:
    @classmethod
    def add_i32(cls, operand1: c_int32, operand2: c_int32) -> c_int32: ...


ctx = PyASTContext()
ctx.register_type(c_int32, builtin.i32)
ctx.register_function(Adder.add_i32, arith.AddiOp)


@ctx.parse_program
def test_add(x: c_int32, y: c_int32) -> c_int32:
    return Adder.add_i32(x, operand2=y)


print(test_add.module)
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   func.func @test_add(%x : i32, %y : i32) -> i32 {
# CHECK-NEXT:     %0 = arith.addi %x, %y : i32
# CHECK-NEXT:     func.return %0 : i32
# CHECK-NEXT:   }
# CHECK-NEXT: }


# CHECK-NEXT: Classmethod arguments must be declared variables.
@ctx.parse_program
def test_args():
    return Adder.add_i32(1, 2)  # pyright: ignore[reportArgumentType]


try:
    test_args.module
except CodeGenerationException as e:
    print(e.msg)


# ================================================= #
# Disable the desymref pass for the remaining tests #
# ================================================= #
ctx.post_transforms = []


# CHECK-NEXT: Classmethod arguments must be declared variables.
@ctx.parse_program
def test_more_args():
    return Adder.add_i32(operand1=1, operand2=2)  # pyright: ignore[reportArgumentType]


try:
    test_more_args.module
except CodeGenerationException as e:
    print(e.msg)


class Class:
    @classmethod
    def method(cls):
        pass


# CHECK-NEXT: Classmethod 'Class.method' is not registered.
@ctx.parse_program
def test_unregistered():
    return Class.method()  # noqa: F821


try:
    test_unregistered.module
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
