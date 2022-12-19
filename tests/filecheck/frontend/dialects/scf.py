# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32

p = FrontendProgram()
with CodeContext(p):

    #      CHECK: %2 : !i1 = symref.fetch() ["symbol" = @cond]
    # CHECK-NEXT: scf.if(%2 : !i1) {
    # CHECK-NEXT:   %3 : !i32 = arith.constant() ["value" = 1 : !i32]
    # CHECK-NEXT:   symref.update(%3 : !i32) ["symbol" = @a]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_1(cond: i1):
        a: i32 = 0
        if cond:
            a = 1

    #      CHECK: %6 : !i1 = symref.fetch() ["symbol" = @cond]
    # CHECK-NEXT: scf.if(%6 : !i1) {
    # CHECK-NEXT:   %7 : !i32 = arith.constant() ["value" = 1 : !i32]
    # CHECK-NEXT:   symref.update(%7 : !i32) ["symbol" = @a]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %8 : !i32 = arith.constant() ["value" = 2 : !i32]
    # CHECK-NEXT:   symref.update(%8 : !i32) ["symbol" = @a]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } 
    def test_if_2(cond: i1):
        a: i32 = 0
        if cond:
            a = 1
        else:
            a = 2
    
    #      CHECK: %12 : !i1 = symref.fetch() ["symbol" = @cond1]
    # CHECK-NEXT: scf.if(%12 : !i1) {
    # CHECK-NEXT:   %13 : !i32 = arith.constant() ["value" = 1 : !i32]
    # CHECK-NEXT:   symref.update(%13 : !i32) ["symbol" = @a]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %14 : !i1 = symref.fetch() ["symbol" = @cond2]
    # CHECK-NEXT:   scf.if(%14 : !i1) {
    # CHECK-NEXT:     %15 : !i32 = arith.constant() ["value" = 2 : !i32]
    # CHECK-NEXT:     symref.update(%15 : !i32) ["symbol" = @a]
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   } {
    # CHECK-NEXT:     %16 : !i32 = arith.constant() ["value" = 3 : !i32]
    # CHECK-NEXT:     symref.update(%16 : !i32) ["symbol" = @a]
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   }
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_3(cond1: i1, cond2: i1):
        a: i32 = 0
        if cond1:
            a = 1
        elif cond2:
            a = 2
        else:
            a = 3
    
    #      CHECK: %18 : !i1 = symref.fetch() ["symbol" = @cond]
    # CHECK-NEXT: %19 : !i32 = scf.if(%18 : !i1) {
    # CHECK-NEXT:   %20 : !i32 = arith.constant() ["value" = 1 : !i32]
    # CHECK-NEXT:   scf.yield(%20 : !i32)
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %21 : !i32 = arith.constant() ["value" = 2 : !i32]
    # CHECK-NEXT:   scf.yield(%21 : !i32)
    # CHECK-NEXT: }
    # CHECK-NEXT: symref.update(%19 : !i32) ["symbol" = @a]
    def test_if_4(cond: i1):
        a: i32 = 1 if cond else 2

p.compile(desymref=False)
print(p.xdsl())
