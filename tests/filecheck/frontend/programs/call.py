# RUN: python %s | filecheck %s

from ctypes import c_int32

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram


def add_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...


p = FrontendProgram()
p.register_type(c_int32, builtin.i32)
p.register_function(add_i32, arith.AddiOp)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @test_add(%x : i32, %y : i32) -> i32 {
    # CHECK-NEXT:     %0 = arith.addi %x, %y : i32
    # CHECK-NEXT:     func.return %0 : i32
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def test_add(x: c_int32, y: c_int32) -> c_int32:
        return add_i32(x, operand2=y)


p.compile()
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK-NEXT: Function arguments must be declared variables.
        def test_args():
            return add_i32(1, 2)  # pyright: ignore[reportArgumentType]

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK-NEXT: Function arguments must be declared variables.
        def test_args():
            return add_i32(operand1=1, operand2=2)  # pyright: ignore[reportArgumentType]

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)


def func():
    pass


with CodeContext(p):

    def test_func():
        return func()  # noqa: F821


try:
    # CHECK-NEXT: Function 'func' is not registered.
    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    # CHECK-NEXT: Function 'func' is not defined in scope.
    del func
    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
