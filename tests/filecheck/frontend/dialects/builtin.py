# RUN: python %s | filecheck %s

from typing import Annotated

from xdsl.dialects.builtin import (
    I1,
    I32,
    I64,
    Float16Type,
    Float32Type,
    Float64Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    Signedness,
    f16,
    f32,
    f64,
    i1,
    i32,
    i64,
)
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

si32 = IntegerType(32, signedness=Signedness.SIGNED)
si64 = IntegerType(64, signedness=Signedness.SIGNED)
ui32 = IntegerType(32, signedness=Signedness.UNSIGNED)
ui64 = IntegerType(64, signedness=Signedness.UNSIGNED)
SI32 = Annotated[IntegerType, si32]
SI64 = Annotated[IntegerType, si64]
UI32 = Annotated[IntegerType, ui32]
UI64 = Annotated[IntegerType, ui64]

p = FrontendProgram()
p.register_type(IntegerAttr[I1], i1)
p.register_type(IntegerAttr[I32], i32)
p.register_type(IntegerAttr[I64], i64)
p.register_type(IntegerAttr[SI32], si32)
p.register_type(IntegerAttr[SI64], si64)
p.register_type(IntegerAttr[UI32], ui32)
p.register_type(IntegerAttr[UI64], ui64)
p.register_type(IndexType, IndexType())
p.register_type(Float16Type, f16)
p.register_type(Float32Type, f32)
p.register_type(Float64Type, f64)
with CodeContext(p):
    # CHECK: @bool(%{{.*}} : i1)
    def bool(x: IntegerAttr[I1]):
        return

    # CHECK: @signless(%{{.*}} : i32, %{{.*}} : i64)
    def signless(x: IntegerAttr[I32], y: IntegerAttr[I64]):
        return

    # CHECK: @unsigned(%{{.*}} : ui32, %{{.*}} : ui64)
    def unsigned(x: IntegerAttr[UI32], y: IntegerAttr[UI64]):
        return

    # CHECK: @signed(%{{.*}} : si32, %{{.*}} : si64)
    def signed(x: IntegerAttr[SI32], y: IntegerAttr[SI64]):
        return

    # CHECK: @indexed(%{{.*}} : index)
    def indexed(x: IndexType):
        return

    # CHECK: @fp(%{{.*}} : f16, %{{.*}} : f32, %{{.*}} : f64)
    def fp(x: Float16Type, y: Float32Type, z: Float64Type):
        return


p.compile(desymref=False)
print(p.textual_format())
