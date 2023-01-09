# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
        
    # CHECK-NOT: symref.declare() ["sym_name" = "a"]
    # CHECK-NOT: symref.update(%{{.*}} : !i64) ["symbol" = @a]
    # CHECK-NOT: %{{.*}} : !i64 = symref.fetch() ["symbol" = @a]
    # CHECK-NOT: symref.declare() ["sym_name" = "b"]
    # CHECK-NOT: symref.update(%{{.*}} : !i64) ["symbol" = @b]
    # CHECK-NOT: %{{.*}} : !i64 = symref.fetch() ["symbol" = @b]
    # CHECK-NOT: symref.declare() ["sym_name" = "c"]
    # CHECK-NOT: symref.update(%{{.*}} : !i64) ["symbol" = @c]
    # CHECK-NOT: %{{.*}} : !i64 = symref.fetch() ["symbol" = @c]
    # CHECK-NOT: symref.declare() ["sym_name" = "d"]
    # CHECK-NOT: symref.update(%{{.*}} : !i64) ["symbol" = @d]
    # CHECK-NOT: %{{.*}} : !i64 = symref.fetch() ["symbol" = @d]

    # CHECK: %0 : !i64 = arith.constant() ["value" = 0 : !i64]
    # CHECK: func.return(%0 : !i64)
    def simple() -> int:
        a: int = 0
        b: int = 1
        c: int = 2
        d: int = 2
        a = a
        b = a
        c = d
        d = a
        b = a
        c = b
        return c

p.compile()
print(p.xdsl())

