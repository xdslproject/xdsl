# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    #      CHECK: %0 : !i64 = arith.constant() ["value" = 0 : !i64]
    # CHECK-NEXT: %1 : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT: %2 : !i64 = affine.for(%0 : !i64) ["lower_bound" = 0 : !index, "upper_bound" = 100 : !index, "step" = 1 : !index] {
    # CHECK-NEXT: ^0(%3 : !index, %4 : !i64):
    # CHECK-NEXT:   %5 : !i64 = arith.addi(%4 : !i64, %1 : !i64)
    # CHECK-NEXT:   affine.yield(%5 : !i64)
    # CHECK-NEXT: }
    # CHECK-NEXT: func.return(%2 : !i64)
    def sum() -> int:
        s = 0
        inc = 1
        for i in range(100):
            s = s + inc
        return s
    
    #      CHECK: %6 : !i64 = arith.constant() ["value" = 0 : !i64]
    # CHECK-NEXT: %7 : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT: %8 : !i64 = arith.constant() ["value" = 2 : !i64]
    #      CHECK: %10 : !i64, %11 : !i64 = affine.for(%6 : !i64, %8 : !i64) ["lower_bound" = 0 : !index, "upper_bound" = 100 : !index, "step" = 1 : !index] {
    # CHECK-NEXT: ^1(%12 : !index, %13 : !i64, %14 : !i64):
    #      CHECK:   %16 : !i64 = arith.constant() ["value" = 3 : !i64]
    # CHECK-NEXT:   %17 : !i64 = arith.addi(%7 : !i64, %16 : !i64)
    # CHECK-NEXT:   affine.yield(%7 : !i64, %17 : !i64)
    # CHECK-NEXT: }
    # CHECK-NEXT: func.return(%11 : !i64)
    def multiple() -> int:
        a = 0
        b = 1
        c = 2
        d = 4
        for i in range(100):
            a = 1000000
            a = b
            c = a + 3
        return c

p.compile()
print(p.xdsl())
