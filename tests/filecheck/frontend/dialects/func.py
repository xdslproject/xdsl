# RUN: python %s | filecheck %s

from xdsl.dialects import bigint
from xdsl.frontend.pyast.context import PyASTContext

ctx = PyASTContext(post_transforms=[])
ctx.register_type(int, bigint.bigint)


# CHECK:      func.func @f1(%{{.*}} : !bigint.bigint) {
# CHECK-NEXT:   symref.declare "x"
# CHECK-NEXT:   symref.update @x = %{{.*}} : !bigint.bigint
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@ctx.parse_program
def f1(x: int):
    return


print(f1.module)


# CHECK:      func.func @f2() {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@ctx.parse_program
def f2():
    return


print(f2.module)


# CHECK:      func.func @f3(%{{.*}} : !bigint.bigint) -> !bigint.bigint {
# CHECK-NEXT:   symref.declare "x"
# CHECK-NEXT:   symref.update @x = %{{.*}} : !bigint.bigint
# CHECK-NEXT:   %{{.*}} = symref.fetch @x : !bigint.bigint
# CHECK-NEXT:   func.return %{{.*}} : !bigint.bigint
# CHECK-NEXT: }
@ctx.parse_program
def f3(x: int) -> int:
    return x


print(f3.module)

try:
    # CHECK:      func.func @f4(%{{.*}} : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:   symref.declare "x"
    # CHECK-NEXT:   symref.update @x = %{{.*}} : !bigint.bigint
    # CHECK-NEXT:   %{{.*}} = symref.fetch @x : !bigint.bigint
    # CHECK-NEXT:   func.return %{{.*}} : !bigint.bigint
    # CHECK-NEXT: }
    @ctx.parse_program
    def f4(x: int) -> int:
        return x

    print(f4.module)
except Exception as e:
    print(e)
    exit(1)
