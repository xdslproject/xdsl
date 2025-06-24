# RUN: python %s | filecheck %s

from xdsl.dialects.builtin import I32, IntegerAttr, i32
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(IntegerAttr[I32], i32)
with CodeContext(p):
    # CHECK:      func.func @f1(%{{.*}} : i32) {
    # CHECK-NEXT:   symref.declare "x"
    # CHECK-NEXT:   symref.update @x = %{{.*}} : i32
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def f1(x: IntegerAttr[I32]):
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
    def f3(x: IntegerAttr[I32]) -> IntegerAttr[I32]:
        return x


p.compile(desymref=False)
print(p.textual_format())
