# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32, Module

p = FrontendProgram()
with CodeContext(p):
    with Module():

        #      CHECK: %0 : !i32 = arith.constant() ["value" = 0 : !i32]
        # CHECK-NEXT: %1 : !i32 = arith.constant() ["value" = 1 : !i32]
        # CHECK-NEXT: %2 : !i32 = affine.for(%0 : !i32) ["lower_bound" = 0 : !index, "upper_bound" = 100 : !index, "step" = 1 : !index] {
        # CHECK-NEXT: ^0(%3 : !index, %4 : !i32):
        # CHECK-NEXT:   %5 : !i32 = arith.addi(%4 : !i32, %1 : !i32)
        # CHECK-NEXT:   affine.yield(%5 : !i32)
        # CHECK-NEXT: }
        # CHECK-NEXT: func.return(%2 : !i32)
        def sum() -> i32:
            s: i32 = 0
            inc: i32 = 1
            for i in range(100):
                s = s + inc
            return s
        
        #      CHECK: %6 : !i32 = arith.constant() ["value" = 0 : !i32]
        # CHECK-NEXT: %7 : !i32 = arith.constant() ["value" = 1 : !i32]
        # CHECK-NEXT: %8 : !i32 = arith.constant() ["value" = 2 : !i32]
        #      CHECK: %10 : !i32, %11 : !i32 = affine.for(%6 : !i32, %8 : !i32) ["lower_bound" = 0 : !index, "upper_bound" = 100 : !index, "step" = 1 : !index] {
        # CHECK-NEXT: ^1(%12 : !index, %13 : !i32, %14 : !i32):
        #      CHECK:   %16 : !i32 = arith.constant() ["value" = 3 : !i32]
        # CHECK-NEXT:   %17 : !i32 = arith.addi(%7 : !i32, %16 : !i32)
        # CHECK-NEXT:   affine.yield(%7 : !i32, %17 : !i32)
        # CHECK-NEXT: }
        # CHECK-NEXT: func.return(%11 : !i32)
        def multiple() -> i32:
            a: i32 = 0
            b: i32 = 1
            c: i32 = 2
            d: i32 = 4
            for i in range(100):
                x: i32 = d
                a = 1000000
                a = b
                c = a + 3
            return c

p.compile()
print(p.xdsl())
