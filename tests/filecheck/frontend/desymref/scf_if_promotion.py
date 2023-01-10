# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):

    # CHECK: %1 : !i64 = arith.constant() ["value" = 0 : !i64]
    # CHECK: %2 : !i64 = scf.if(%0 : !i1) {
    # CHECK:   %3 : !i64 = arith.constant() ["value" = 2 : !i64]
    # CHECK:   scf.yield(%3 : !i64)
    # CHECK: } {
    # CHECK:   scf.yield(%1 : !i64)
    # CHECK: }
    # CHECK: func.return(%2 : !i64)
    def single_if(cond: bool) -> int:
        a: int = 0
        if cond:
            a = 2
        return a

    # CHECK: %6 : !i64 = arith.constant() ["value" = 7 : !i64]
    # CHECK: %7 : !i64 = scf.if(%4 : !i1) {
    # CHECK:   scf.yield(%6 : !i64)
    # CHECK: } {
    # CHECK:   %8 : !i64 = arith.constant() ["value" = 10 : !i64]
    # CHECK:   scf.yield(%8 : !i64)
    # CHECK: }
    # CHECK: func.return(%7 : !i64)
    def if_else(cond: bool) -> int:
        a: int = 0
        c: int = 7
        if cond:
            a = c
        else:
            b: int = 10
            a = b
        return a
    
    # CHECK: %12 : !i64 = arith.constant() ["value" = 0 : !i64]
    # CHECK: %13 : !i64 = scf.if(%9 : !i1) {
    # CHECK:   %14 : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK:   scf.yield(%14 : !i64)
    # CHECK: } {
    # CHECK:   %15 : !i64 = scf.if(%10 : !i1) {
    # CHECK:     %16 : !i64 = arith.constant() ["value" = 2 : !i64]
    # CHECK:     scf.yield(%16 : !i64)
    # CHECK:   } {
    # CHECK:     %18 : !i64 = scf.if(%11 : !i1) {
    # CHECK:       %19 : !i64 = arith.constant() ["value" = 3 : !i64]
    # CHECK:       scf.yield(%19 : !i64)
    # CHECK:     } {
    # CHECK:       scf.yield(%12 : !i64)
    # CHECK:     }
    # CHECK:     scf.yield(%18 : !i64)
    # CHECK:   }
    # CHECK:   scf.yield(%15 : !i64)
    # CHECK: }
    # CHECK: func.return(%13 : !i64)
    def elifs(cond1: bool, cond2: bool, cond3: bool) -> int:
        a: int = 0
        if cond1:
            a = 1
        elif cond2:
            b: int = 2
            a = b
            b = 100000
        elif cond3:
            c: int = 3
            a = c
            c = a
        return a
    
    #      CHECK: ^{{.*}}(%{{.*}} : !i1, %21 : !i64, %22 : !i64):
    # CHECK-NEXT:   %23 : !i64 = scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:     scf.yield(%21 : !i64)
    # CHECK-NEXT:   } {
    # CHECK-NEXT:     scf.yield(%22 : !i64)
    # CHECK-NEXT:   }
    # CHECK-NEXT:   func.return(%23 : !i64)
    # CHECK-NEXT: }
    def test_if_expr(cond: bool, b: int, c: int) -> int:
        a: int = b if cond else c
        return a

p.compile()
print(p.xdsl())
