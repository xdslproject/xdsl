# RUN: python %s | filecheck %s

from ctypes import c_float, c_int32, c_int64, c_size_t

from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(c_int32, builtin.i32)
p.register_type(c_int64, builtin.i64)
p.register_type(c_size_t, builtin.IndexType())
p.register_type(bool, builtin.i1)
p.register_type(c_float, builtin.f32)
p.register_type(float, builtin.f64)
with CodeContext(p):
    # CHECK: @boolean(%{{.*}} : i1)
    def boolean(x: bool):
        return

    # CHECK: @signless(%{{.*}} : i32, %{{.*}} : i64)
    def signless(x: c_int32, y: c_int64):
        return

    # CHECK: @indexed(%{{.*}} : index)
    def indexed(x: c_size_t):
        return

    # CHECK: @fp(%{{.*}} : f32, %{{.*}} : f64)
    def fp(x: c_float, y: float):
        return


p.compile(desymref=False)
print(p.textual_format())
