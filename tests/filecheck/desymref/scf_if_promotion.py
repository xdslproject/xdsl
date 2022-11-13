# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, Module

p = FrontendProgram()
with CodeContext(p):
    with Module():

        # CHECK: %1 : !i32 = arith.constant() ["value" = 0 : !i32]
        # CHECK: %2 : !i32 = scf.if(%0 : !i1) {
        # CHECK:   %3 : !i32 = arith.constant() ["value" = 2 : !i32]
        # CHECK:   scf.yield(%3 : !i32)
        # CHECK: } {
        # CHECK:   scf.yield(%1 : !i32)
        # CHECK: }
        # CHECK: func.return(%2 : !i32)
        def single_if(cond: i1) -> i32:
            a: i32 = 0
            if cond:
                a = 2
            return a

        # CHECK: %6 : !i32 = arith.constant() ["value" = 7 : !i32]
        # CHECK: %7 : !i32 = scf.if(%4 : !i1) {
        # CHECK:   scf.yield(%6 : !i32)
        # CHECK: } {
        # CHECK:   %8 : !i32 = arith.constant() ["value" = 10 : !i32]
        # CHECK:   scf.yield(%8 : !i32)
        # CHECK: }
        # CHECK: func.return(%7 : !i32)
        def if_else(cond: i1) -> i32:
            a: i32 = 0
            c: i32 = 7
            if cond:
                a = c
            else:
                b: i32 = 10
                a = b
            return a
        
        # CHECK: %12 : !i32 = arith.constant() ["value" = 0 : !i32]
        # CHECK: %13 : !i32 = scf.if(%9 : !i1) {
        # CHECK:   %14 : !i32 = arith.constant() ["value" = 1 : !i32]
        # CHECK:   scf.yield(%14 : !i32)
        # CHECK: } {
        # CHECK:   %15 : !i32 = scf.if(%10 : !i1) {
        # CHECK:     %16 : !i32 = arith.constant() ["value" = 2 : !i32]
        # CHECK:     scf.yield(%16 : !i32)
        # CHECK:   } {
        # CHECK:     %18 : !i32 = scf.if(%11 : !i1) {
        # CHECK:       %19 : !i32 = arith.constant() ["value" = 3 : !i32]
        # CHECK:       scf.yield(%19 : !i32)
        # CHECK:     } {
        # CHECK:       scf.yield(%12 : !i32)
        # CHECK:     }
        # CHECK:     scf.yield(%18 : !i32)
        # CHECK:   }
        # CHECK:   scf.yield(%15 : !i32)
        # CHECK: }
        # CHECK: func.return(%13 : !i32)
        def elifs(cond1: i1, cond2: i1, cond3: i1) -> i32:
            a: i32 = 0
            if cond1:
                a = 1
            elif cond2:
                b: i32 = 2
                a = b
                b = 100000
            elif cond3:
                c: i32 = 3
                a = c
                c = a
            return a

p.compile()
p.desymref()
print(p)