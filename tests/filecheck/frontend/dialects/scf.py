# RUN: python %s | filecheck %s

from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, f32

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      %{{.*}} : !i32 = scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   %{{.*}} : !i32 = symref.fetch() ["symbol" = @x]
    # CHECK-NEXT:   scf.yield(%{{.*}} : !i32)
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %{{.*}} : !i32 = symref.fetch() ["symbol" = @y]
    # CHECK-NEXT:   scf.yield(%{{.*}} : !i32)
    # CHECK-NEXT: }
    def test_if_expr(cond: i1, x: i32, y: i32) -> i32:
        return x if cond else y

    # CHECK:      scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_I(cond: i1):
        if cond:
            pass
        else:
            pass
        return

    # CHECK:       %{{.*}} : !i1 = symref.fetch() ["symbol" = @a]
    # CHECK-NEXT:  scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   %{{.*}} : !i1 = symref.fetch() ["symbol" = @b]
    # CHECK-NEXT:   scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   } {
    # CHECK-NEXT:     %{{.*}} : !i1 = symref.fetch() ["symbol" = @c]
    # CHECK-NEXT:     scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:       scf.yield()
    # CHECK-NEXT:     } {
    # CHECK-NEXT:       scf.yield()
    # CHECK-NEXT:     }
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   }
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_II(a: i1, b: i1, c: i1):
        if a:
            pass
        elif b:
            pass
        elif c:
            pass
        return

    # CHECK:      %{{.*}} : !i1 = symref.fetch() ["symbol" = @cond]
    # CHECK-NEXT: scf.if(%{{.*}} : !i1) {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: } {
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_if_III(cond: i1):
        if cond:
            pass
        return


p.compile(desymref=False)
print(p.xdsl())


try:
    with CodeContext(p):
        # CHECK: Expected the same types for if expression, but got i32 and f32.
        def test_type_mismatch_in_if_expr(cond: i1, x: i32, y: f32) -> i32:
            return x if cond else y  # type: ignore

    p.compile(desymref=False)
    print(p.xdsl())
except CodeGenerationException as e:
    print(e.msg)
