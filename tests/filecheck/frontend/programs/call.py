# RUN: python %s | filecheck %s

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram


def add_bigint(operand1: int, operand2: int) -> int: ...


p1 = FrontendProgram()
p1.register_type(int, builtin.i32)
p1.register_function(add_bigint, arith.AddiOp)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%x : i32, %y : i32) -> i32 {
    # CHECK-NEXT:     %0 = arith.addi %x, %y : i32
    # CHECK-NEXT:     func.return %0 : i32
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: int, y: int) -> int:
        return add_bigint(x, operand2=y)


p1.compile()
print(p1.textual_format())
