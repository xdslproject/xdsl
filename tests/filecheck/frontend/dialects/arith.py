# RUN: python %s | filecheck %s

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    I1,
    I32,
    I64,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    f16,
    f32,
    f64,
    i1,
    i32,
    i64,
)
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(IntegerAttr[I1], i1)
p.register_type(IntegerAttr[I32], i32)
p.register_type(IntegerAttr[I64], i64)
p.register_type(FloatAttr[Float16Type], f16)
p.register_type(FloatAttr[Float32Type], f32)
p.register_type(FloatAttr[Float64Type], f64)
p.register_function(IntegerAttr.__add__, arith.AddiOp)
p.register_function(IntegerAttr.__and__, arith.AndIOp)
p.register_function(IntegerAttr.__mul__, arith.MuliOp)
p.register_function(IntegerAttr.__sub__, arith.SubiOp)
p.register_function(IntegerAttr.__lshift__, arith.ShLIOp)
p.register_function(IntegerAttr.__rshift__, arith.ShRSIOp)
p.register_function(IntegerAttr.__eq__, lambda x, y: arith.CmpiOp(x, y, "eq"))
p.register_function(IntegerAttr.__ne__, lambda x, y: arith.CmpiOp(x, y, "ne"))
p.register_function(IntegerAttr.__lt__, lambda x, y: arith.CmpiOp(x, y, "slt"))
p.register_function(IntegerAttr.__le__, lambda x, y: arith.CmpiOp(x, y, "sle"))
p.register_function(IntegerAttr.__gt__, lambda x, y: arith.CmpiOp(x, y, "sgt"))
p.register_function(IntegerAttr.__ge__, lambda x, y: arith.CmpiOp(x, y, "sge"))
p.register_function(FloatAttr.__add__, arith.AddfOp)
p.register_function(FloatAttr.__sub__, arith.SubfOp)
p.register_function(FloatAttr.__mul__, arith.MulfOp)

with CodeContext(p):
    # CHECK: arith.addi %{{.*}}, %{{.*}} : i32
    def test_addi_overload(
        a: IntegerAttr[I32], b: IntegerAttr[I32]
    ) -> IntegerAttr[I32]:
        return a + b

    # CHECK: arith.andi %{{.*}}, %{{.*}} : i64
    def test_andi_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I64]:
        return a & b

    # CHECK: arith.muli %{{.*}}, %{{.*}} : i32
    def test_muli_overload(
        a: IntegerAttr[I32], b: IntegerAttr[I32]
    ) -> IntegerAttr[I32]:
        return a * b

    # CHECK: arith.subi %{{.*}}, %{{.*}} : i64
    def test_subi_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I64]:
        return a - b

    # CHECK: arith.shli %{{.*}}, %{{.*}} : i32
    def test_shli_overload(
        a: IntegerAttr[I32], b: IntegerAttr[I32]
    ) -> IntegerAttr[I32]:
        return a << b

    # CHECK: arith.shrsi %{{.*}}, %{{.*}} : i64
    def test_shrsi_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I64]:
        return a >> b

    # CHECK:  arith.cmpi eq, %{{.*}}, %{{.*}} : i32
    def test_cmpi_eq_overload(
        a: IntegerAttr[I32], b: IntegerAttr[I32]
    ) -> IntegerAttr[I1]:
        return a == b

    # CHECKNO: arith.cmpi sle, %{{.*}}, %{{.*}} : i64
    def test_cmpi_le_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I1]:
        return a <= b

    # CHECKNO: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
    def test_cmpi_lt_overload(
        a: IntegerAttr[I32], b: IntegerAttr[I32]
    ) -> IntegerAttr[I1]:
        return a < b

    # CHECKNO: arith.cmpi sge, %{{.*}}, %{{.*}} : i64
    def test_cmpi_ge_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I1]:
        return a >= b

    # CHECKNO: arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
    def test_cmpi_gt_overload(
        a: IntegerAttr[I32], b: IntegerAttr[I32]
    ) -> IntegerAttr[I1]:
        return a > b

    # CHECKNO: arith.cmpi ne, %{{.*}}, %{{.*}} : i64
    def test_cmpi_ne_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I1]:
        return a != b

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f16
    def test_addf_overload_f16(
        a: FloatAttr[Float16Type], b: FloatAttr[Float16Type]
    ) -> FloatAttr[Float16Type]:
        return a + b

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f32
    def test_addf_overload_f32(
        a: FloatAttr[Float32Type], b: FloatAttr[Float32Type]
    ) -> FloatAttr[Float32Type]:
        return a + b

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f64
    def test_addf_overload_f64(
        a: FloatAttr[Float64Type], b: FloatAttr[Float64Type]
    ) -> FloatAttr[Float64Type]:
        return a + b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f16
    def test_subf_overload_f16(
        a: FloatAttr[Float16Type], b: FloatAttr[Float16Type]
    ) -> FloatAttr[Float16Type]:
        return a - b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f32
    def test_subf_overload_f32(
        a: FloatAttr[Float32Type], b: FloatAttr[Float32Type]
    ) -> FloatAttr[Float32Type]:
        return a - b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f64
    def test_subf_overload_f64(
        a: FloatAttr[Float64Type], b: FloatAttr[Float64Type]
    ) -> FloatAttr[Float64Type]:
        return a - b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f16
    def test_mulf_overload_f16(
        a: FloatAttr[Float16Type], b: FloatAttr[Float16Type]
    ) -> FloatAttr[Float16Type]:
        return a * b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f32
    def test_mulf_overload_f32(
        a: FloatAttr[Float32Type], b: FloatAttr[Float32Type]
    ) -> FloatAttr[Float32Type]:
        return a * b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
    def test_mulf_overload_f64(
        a: FloatAttr[Float64Type], b: FloatAttr[Float64Type]
    ) -> FloatAttr[Float64Type]:
        return a * b


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Binary operation 'FloorDiv' is not supported by type 'FloatAttr' which does not overload '__floordiv__'.
        def test_missing_floordiv_overload_f64(
            a: FloatAttr[Float64Type], b: FloatAttr[Float64Type]
        ) -> FloatAttr[Float64Type]:
            # We expect the type error here, since the function doesn't exist on f64
            return a // b

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Comparison operation 'In' is not supported by type 'f64' which does not overload '__contains__'.
        def test_missing_contains_overload_f64(
            a: FloatAttr[Float64Type], b: FloatAttr[Float64Type]
        ) -> FloatAttr[Float64Type]:
            # We expect the type error here, since the function doesn't exist on f64
            return a in b

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
