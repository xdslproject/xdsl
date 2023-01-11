# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import block

p = FrontendProgram()
with CodeContext(p):

    #      CHECK: func.func() ["sym_name" = "blocks"
    # CHECK-NEXT: ^0(%{{.*}} : !i64, %{{.*}} : !i64, %{{.*}} : !i64):

    #      CHECK:   cf.br(%{{.*}} : !i64) (^1)
    # CHECK-NEXT: ^1(%{{.*}} : !i64):

    #      CHECK:   cf.br(%{{.*}} : !i64, %{{.*}} : !i64) (^2)
    # CHECK-NEXT: ^2(%{{.*}} : !i64, %{{.*}} : !i64):

    #      CHECK:   cf.br() (^3)
    # CHECK-NEXT: ^3:
    # CHECK-NEXT:   func.return()
    def blocks(a: int, b: int, c: int):
        bb1(a)

        @block
        def bb1(x: int):
            t1: int = x + b
            t2: int = x + c
            bb2(t1, t2)

        @block
        def bb2(x: int, y: int):
            t3: int = x + y
            bb3()

        @block
        def bb3():
            return

p.compile(desymref=False)
print(p.xdsl())
