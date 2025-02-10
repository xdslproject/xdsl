# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.dialects.builtin import i32
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      func.func @f1(%{{.*}} : i32) {
    # CHECK-NEXT:   symref.declare "x"
    # CHECK-NEXT:   symref.update @x = %{{.*}} : i32
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def f1(x: i32):
        return

    # CHECK:      func.func @f2() {
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def f2():
        return

    # CHECK:      func.func @f3(%{{.*}} : i32) -> i32 {
    # CHECK-NEXT:   symref.declare "x"
    # CHECK-NEXT:   symref.update @x = %{{.*}} : i32
    # CHECK-NEXT:   %{{.*}} = symref.fetch @x : i32
    # CHECK-NEXT:   func.return %{{.*}} : i32
    # CHECK-NEXT: }
    def f3(x: i32) -> i32:
        return x


p.compile(desymref=False)
print(p.textual_format())
