# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, i64, f16, f32, f64
from xdsl.frontend.dialects import arith
from xdsl.frontend.dialects.arith import andi, subi, cmpi, addf, subf, mulf

p = FrontendProgram()
with CodeContext(p):

    # CHECK: %{{.*}} : !i64 = arith.addi(%{{.*}} : !i64, %{{.*}} : !i64)
    def test_addi(a: i64, b: i64) -> i64:
        return arith.addi(a, b)
    
    # CHECK: %{{.*}} : !i32 = arith.addi(%{{.*}} : !i32, %{{.*}} : !i32)
    def test_addi_overload(a: i32, b: i32) -> i32:
        return a + b


    # CHECK: %{{.*}} : !i64 = arith.subi(%{{.*}} : !i64, %{{.*}} : !i64)
    def test_subi(a: i64, b: i64) -> i64:
        return subi(a, b)

    # CHECK: %{{.*}} : !i32 = arith.subi(%{{.*}} : !i32, %{{.*}} : !i32)
    def test_subi_overload(a: i32, b: i32) -> i32:
        return a - b


    # CHECK: %{{.*}} : !i64 = arith.muli(%{{.*}} : !i64, %{{.*}} : !i64)
    def test_muli(a: i64, b: i64) -> i64:
        return arith.muli(a, b)

    # CHECK: %{{.*}} : !i32 = arith.muli(%{{.*}} : !i32, %{{.*}} : !i32)
    def test_muli_overload(a: i32, b: i32) -> i32:
        return a * b


    # CHECK: %{{.*}} : !i64 = arith.andi(%{{.*}} : !i64, %{{.*}} : !i64)
    def test_andi(a: i64, b: i64) -> i64:
        return andi(a, b)

    # CHECK: %{{.*}} : !i32 = arith.andi(%{{.*}} : !i32, %{{.*}} : !i32)
    def test_andi_overload(a: i32, b: i32) -> i32:
        return a & b


    # CHECK: %{{.*}} : !i64 = arith.shrsi(%{{.*}} : !i64, %{{.*}} : !i64)
    def test_shrsi(a: i64, b: i64) -> i64:
        return arith.shrsi(a, b)

    # CHECK: %{{.*}} : !i32 = arith.shrsi(%{{.*}} : !i32, %{{.*}} : !i32)
    def test_shrsi_overload(a: i32, b: i32) -> i32:
        return a >> b
    

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 0 : !i64]
    def test_cmpi_eq(a: i64, b: i64) -> i1:
        return arith.cmpi(a, b, "eq")

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 1 : !i64]
    def test_cmpi_ne(a: i64, b: i64) -> i1:
        return cmpi(a, b, "ne")
    
    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 3 : !i64]
    def test_cmpi_le(a: i64, b: i64) -> i1:
        return arith.cmpi(a, b, "sle")

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 2 : !i64]
    def test_cmpi_lt(a: i64, b: i64) -> i1:
        return cmpi(a, b, "slt")

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 5 : !i64]
    def test_cmpi_ge(a: i64, b: i64) -> i1:
        return arith.cmpi(a, b, "sge")

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 4 : !i64]
    def test_cmpi_gt(a: i64, b: i64) -> i1:
        return cmpi(a, b, "sgt")

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i32, %{{.*}} : !i32) ["predicate" = 0 : !i64]
    def test_cmpi_eq_overload(a: i32, b: i32) -> i1:
        return a == b
    
    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i32, %{{.*}} : !i32) ["predicate" = 1 : !i64]
    def test_cmpi_ne_overload(a: i32, b: i32) -> i1:
        return a != b
    
    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i32, %{{.*}} : !i32) ["predicate" = 3 : !i64]
    def test_cmpi_le_overload(a: i32, b: i32) -> i1:
        return a <= b
    
    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i32, %{{.*}} : !i32) ["predicate" = 2 : !i64]
    def test_cmpi_lt_overload(a: i32, b: i32) -> i1:
        return a < b

    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i32, %{{.*}} : !i32) ["predicate" = 5 : !i64]
    def test_cmpi_ge_overload(a: i32, b: i32) -> i1:
        return a >= b
    
    # CHECK: %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i32, %{{.*}} : !i32) ["predicate" = 4 : !i64]
    def test_cmpi_gt_overload(a: i32, b: i32) -> i1:
        return a > b


    # CHECK: %{{.*}} : !f16 = arith.addf(%{{.*}} : !f16, %{{.*}} : !f16)
    def test_addf_f16(a: f16, b: f16) -> f16:
        return arith.addf(a, b)
    
    # CHECK: %{{.*}} : !f32 = arith.addf(%{{.*}} : !f32, %{{.*}} : !f32)
    def test_addf_f32(a: f32, b: f32) -> f32:
        return addf(a, b)
    
    # CHECK: %{{.*}} : !f64 = arith.addf(%{{.*}} : !f64, %{{.*}} : !f64)
    def test_addf_f64(a: f64, b: f64) -> f64:
        return arith.addf(a, b)

    # CHECK: %{{.*}} : !f16 = arith.addf(%{{.*}} : !f16, %{{.*}} : !f16)
    def test_addf_overload_f16(a: f16, b: f16) -> f16:
        return a + b
    
    # CHECK: %{{.*}} : !f32 = arith.addf(%{{.*}} : !f32, %{{.*}} : !f32)
    def test_addf_overload_f32(a: f32, b: f32) -> f32:
        return a + b
    
    # CHECK: %{{.*}} : !f64 = arith.addf(%{{.*}} : !f64, %{{.*}} : !f64)
    def test_addf_overload_f64(a: f64, b: f64) -> f64:
        return a + b
    

    # CHECK: %{{.*}} : !f16 = arith.subf(%{{.*}} : !f16, %{{.*}} : !f16)
    def test_subf_f16(a: f16, b: f16) -> f16:
        return arith.subf(a, b)
    
    # CHECK: %{{.*}} : !f32 = arith.subf(%{{.*}} : !f32, %{{.*}} : !f32)
    def test_subf_f32(a: f32, b: f32) -> f32:
        return subf(a, b)
    
    # CHECK: %{{.*}} : !f64 = arith.subf(%{{.*}} : !f64, %{{.*}} : !f64)
    def test_subf_f64(a: f64, b: f64) -> f64:
        return arith.subf(a, b)

    # CHECK: %{{.*}} : !f16 = arith.subf(%{{.*}} : !f16, %{{.*}} : !f16)
    def test_subf_overload_f16(a: f16, b: f16) -> f16:
        return a - b
    
    # CHECK: %{{.*}} : !f32 = arith.subf(%{{.*}} : !f32, %{{.*}} : !f32)
    def test_subf_overload_f32(a: f32, b: f32) -> f32:
        return a - b
    
    # CHECK: %{{.*}} : !f64 = arith.subf(%{{.*}} : !f64, %{{.*}} : !f64)
    def test_subf_overload_f64(a: f64, b: f64) -> f64:
        return a - b
    

    # CHECK: %{{.*}} : !f16 = arith.mulf(%{{.*}} : !f16, %{{.*}} : !f16)
    def test_mulf_f16(a: f16, b: f16) -> f16:
        return arith.mulf(a, b)
    
    # CHECK: %{{.*}} : !f32 = arith.mulf(%{{.*}} : !f32, %{{.*}} : !f32)
    def test_mulf_f32(a: f32, b: f32) -> f32:
        return mulf(a, b)
    
    # CHECK: %{{.*}} : !f64 = arith.mulf(%{{.*}} : !f64, %{{.*}} : !f64)
    def test_mulf_f64(a: f64, b: f64) -> f64:
        return arith.mulf(a, b)

    # CHECK: %{{.*}} : !f16 = arith.mulf(%{{.*}} : !f16, %{{.*}} : !f16)
    def test_mulf_overload_f16(a: f16, b: f16) -> f16:
        return a * b
    
    # CHECK: %{{.*}} : !f32 = arith.mulf(%{{.*}} : !f32, %{{.*}} : !f32)
    def test_mulf_overload_f32(a: f32, b: f32) -> f32:
        return a * b
    
    # CHECK: %{{.*}} : !f64 = arith.mulf(%{{.*}} : !f64, %{{.*}} : !f64)
    def test_mulf_overload_f64(a: f64, b: f64) -> f64:
        return a * b

p.compile()
print(p.xdsl())
