# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.dialects.builtin import i32
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%0 : i32, %1 : i32, %2 : i32) -> i32 {
    # CHECK-NEXT:     %3 = arith.muli %1, %2 : i32
    # CHECK-NEXT:     %4 = arith.addi %0, %3 : i32
    # CHECK-NEXT:     func.return %4 : i32
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: i32, y: i32, z: i32) -> i32:
        return x + y * z


p.compile()
print(p.textual_format())
