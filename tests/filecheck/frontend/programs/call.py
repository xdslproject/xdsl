# RUN: python %s | filecheck %s

from ctypes import c_int32

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram


def add_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...


p1 = FrontendProgram()
p1.register_type(c_int32, builtin.i32)
p1.register_function(add_i32, arith.AddiOp)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%x : i32, %y : i32) -> i32 {
    # CHECK-NEXT:     %0 = arith.addi %x, %y : i32
    # CHECK-NEXT:     func.return %0 : i32
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: c_int32, y: c_int32) -> c_int32:
        return add_i32(x, operand2=y)


p1.compile()
print(p1.textual_format())
