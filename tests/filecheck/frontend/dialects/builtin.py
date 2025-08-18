# RUN: python %s | filecheck %s

from ctypes import c_float, c_int32, c_int64, c_size_t

from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import PyASTContext

ctx = PyASTContext(post_transforms=[])
ctx.register_type(c_int32, builtin.i32)
ctx.register_type(c_int64, builtin.i64)
ctx.register_type(c_size_t, builtin.IndexType())
ctx.register_type(bool, builtin.i1)
ctx.register_type(c_float, builtin.f32)
ctx.register_type(float, builtin.f64)


# CHECK: @boolean(%{{.*}} : i1)
@ctx.parse_program
def boolean(x: bool):
    return


print(boolean.module)


# CHECK: @signless(%{{.*}} : i32, %{{.*}} : i64)
@ctx.parse_program
def signless(x: c_int32, y: c_int64):
    return


print(signless.module)


# CHECK: @indexed(%{{.*}} : index)
@ctx.parse_program
def indexed(x: c_size_t):
    return


print(indexed.module)


# CHECK: @fp(%{{.*}} : f32, %{{.*}} : f64)
@ctx.parse_program
def fp(x: c_float, y: float):
    return


print(fp.module)
