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
        def unused():
            a: i32 = 0
            for i in range(10):
                b: i32 = 0
                c: i32 = 1
                d: i32 = 2
                b = c + d

p.compile()
p.desymref()
print(p)
