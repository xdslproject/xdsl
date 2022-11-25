# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32, Module

p = FrontendProgram()
with CodeContext(p):
    with Module():
        
        # CHECK-NOT: symref.declare() ["sym_name" = "a"]
        # CHECK-NOT: symref.update(%{{.*}} : !i32) ["symbol" = @a]
        # CHECK-NOT: %{{.*}} : !i32 = symref.fetch() ["symbol" = @a]
        # CHECK-NOT: symref.declare() ["sym_name" = "b"]
        # CHECK-NOT: symref.update(%{{.*}} : !i32) ["symbol" = @b]
        # CHECK-NOT: %{{.*}} : !i32 = symref.fetch() ["symbol" = @b]
        # CHECK-NOT: symref.declare() ["sym_name" = "c"]
        # CHECK-NOT: symref.update(%{{.*}} : !i32) ["symbol" = @c]
        # CHECK-NOT: %{{.*}} : !i32 = symref.fetch() ["symbol" = @c]
        # CHECK-NOT: symref.declare() ["sym_name" = "d"]
        # CHECK-NOT: symref.update(%{{.*}} : !i32) ["symbol" = @d]
        # CHECK-NOT: %{{.*}} : !i32 = symref.fetch() ["symbol" = @d]

        # CHECK: %0 : !i32 = arith.constant() ["value" = 0 : !i32]
        # CHECK: func.return(%0 : !i32)
        def simple() -> i32:
            a: i32 = 0
            b: i32 = 1
            c: i32 = 2
            d: i32 = 2
            a = a
            b = a
            c = d
            d = a
            b = a
            c = b
            return c

p.compile()
p.desymref()
print(p.xdsl())

