# RUN: python %s | filecheck %s

from xdsl.dialects import bigint
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, bigint.bigint)
with CodeContext(p):
    # CHECK:      func.func @f1(%{{.*}} : !bigint.bigint) {
    # CHECK-NEXT:   symref.declare "x"
    # CHECK-NEXT:   symref.update @x = %{{.*}} : !bigint.bigint
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def f1(x: int):
        return

    # CHECK:      func.func @f2() {
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def f2():
        return

    # CHECK:      func.func @f3(%{{.*}} : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:   symref.declare "x"
    # CHECK-NEXT:   symref.update @x = %{{.*}} : !bigint.bigint
    # CHECK-NEXT:   %{{.*}} = symref.fetch @x : !bigint.bigint
    # CHECK-NEXT:   func.return %{{.*}} : !bigint.bigint
    # CHECK-NEXT: }
    def f3(x: int) -> int:
        return x


p.compile(desymref=False)
print(p.textual_format())
