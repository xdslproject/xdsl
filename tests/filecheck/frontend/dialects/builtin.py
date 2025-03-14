# RUN: python %s | filecheck %s


from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.dialects.builtin import (
    f16,
    f32,
    f64,
    i1,
    i32,
    i64,
    index,
    si32,
    si64,
    ui32,
    ui64,
)
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK: @bool(%{{.*}} : i1)
    def bool(x: i1):
        return

    # CHECK: @signless(%{{.*}} : i32, %{{.*}} : i64)
    def signless(x: i32, y: i64):
        return

    # CHECK: @unsigned(%{{.*}} : ui32, %{{.*}} : ui64)
    def unsigned(x: ui32, y: ui64):
        return

    # CHECK: @signed(%{{.*}} : si32, %{{.*}} : si64)
    def signed(x: si32, y: si64):
        return

    # CHECK: @indexed(%{{.*}} : index)
    def indexed(x: index):
        return

    # CHECK: @fp(%{{.*}} : f16, %{{.*}} : f32, %{{.*}} : f64)
    def fp(x: f16, y: f32, z: f64):
        return


p.compile(desymref=False)
print(p.textual_format())
