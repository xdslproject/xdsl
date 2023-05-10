# RUN: python %s | filecheck %s

from typing import Literal, Tuple
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import (
    i1,
    i32,
    i64,
    ui32,
    ui64,
    si32,
    si64,
    index,
    f16,
    f32,
    f64,
)

p = FrontendProgram()
with CodeContext(p):
    #      CHECK: "func.func"()
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : i1)
    def bool(x: i1):
        return

    #      CHECK: "func.func"()
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : i32, %{{.*}} : i64)
    def signless(x: i32, y: i64):
        return

    #      CHECK: "func.func"()
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : ui32, %{{.*}} : ui64)
    def unsigned(x: ui32, y: ui64):
        return

    #      CHECK: "func.func"()
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : si32, %{{.*}} : si64)
    def signed(x: si32, y: si64):
        return

    #      CHECK: "func.func"()
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : index)
    def indexed(x: index):
        return

    #      CHECK: "func.func"()
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : f16, %{{.*}} : f32, %{{.*}} : f64)
    def fp(x: f16, y: f32, z: f64):
        return


p.compile(desymref=False)
print(p.textual_format())
