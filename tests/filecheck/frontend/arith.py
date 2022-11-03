# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32

p = FrontendProgram()
with CodeContext(p):
    # CHECK: %{{.*}} : !i32 = arith.addi(%{{.*}} : !i32, %{{.*}} : !i32)
    def addi(a: i32, b: i32) -> i32:
        return a + b

    # CHECK: %{{.*}} : !i32 = arith.muli(%{{.*}} : !i32, %{{.*}} : !i32)
    def muli(a: i32, b: i32) -> i32:
        return a * b

    # CHECK: %{{.*}} : !i32 = arith.subi(%{{.*}} : !i32, %{{.*}} : !i32)
    def subi(a: i32, b: i32) -> i32:
        return a - b

    # CHECK: %{{.*}} : !i32 = arith.andi(%{{.*}} : !i32, %{{.*}} : !i32)
    def andi(a: i32, b: i32) -> i32:
        return a & b

    # CHECK: %{{.*}} : !i32 = arith.shrsi(%{{.*}} : !i32, %{{.*}} : !i32)
    def shrsi(a: i32, b: i32) -> i32:
        return a >> b

p.compile()
p.desymref()
print(p)
