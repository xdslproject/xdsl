# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):

    #      CHECK: %{{.*}} : !i1 = symref.fetch() ["symbol" = @{{.*}}]
    # CHECK-NEXT: scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_1(cond: bool):
        a: int = 0
        if cond:
            a = 1

    #      CHECK: %{{.*}} : !i1 = symref.fetch() ["symbol" = @{{.*}}]
    # CHECK-NEXT: scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 2 : !i64]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } 
    def test_if_2(cond: bool):
        a: int = 0
        if cond:
            a = 1
        else:
            a = 2
    
    #      CHECK: %{{.*}} : !i1 = symref.fetch() ["symbol" = @{{.*}}]
    # CHECK-NEXT: scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %{{.*}} : !i1 = symref.fetch() ["symbol" = @{{.*}}]
    # CHECK-NEXT:   scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:     %{{.*}} : !i64 = arith.constant() ["value" = 2 : !i64]
    # CHECK-NEXT:     symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   } {
    # CHECK-NEXT:     %{{.*}} : !i64 = arith.constant() ["value" = 3 : !i64]
    # CHECK-NEXT:     symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   }
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_3(cond1: bool, cond2: bool):
        a: int = 0
        if cond1:
            a = 1
        elif cond2:
            a = 2
        else:
            a = 3
    
    #      CHECK: %{{.*}} : !i1 = symref.fetch() ["symbol" = @{{.*}}]
    # CHECK-NEXT: %{{.*}} : !i64 = scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT:   scf.yield(%{{.*}} : !i64)
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 2 : !i64]
    # CHECK-NEXT:   scf.yield(%{{.*}} : !i64)
    # CHECK-NEXT: }
    # CHECK-NEXT: symref.update(%{{.*}} : !i64) ["symbol" = @{{.*}}]
    def test_if_4(cond: bool):
        a: int = 1 if cond else 2

p.compile(desymref=False)
print(p.xdsl())
