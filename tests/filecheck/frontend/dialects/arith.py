# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.dialects.builtin import f16, f32, f64, i1, i32, i64
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK: arith.addi %{{.*}}, %{{.*}} : i32
    def test_addi_overload(a: i32, b: i32) -> i32:
        return a + b

    # CHECK: arith.andi %{{.*}}, %{{.*}} : i64
    def test_andi_overload(a: i64, b: i64) -> i64:
        return a & b

    # CHECK: arith.muli %{{.*}}, %{{.*}} : i32
    def test_muli_overload(a: i32, b: i32) -> i32:
        return a * b

    # CHECK: arith.subi %{{.*}}, %{{.*}} : i64
    def test_subi_overload(a: i64, b: i64) -> i64:
        return a - b

    # CHECK: arith.shli %{{.*}}, %{{.*}} : i32
    def test_shli_overload(a: i32, b: i32) -> i32:
        return a << b

    # CHECK: arith.shrsi %{{.*}}, %{{.*}} : i64
    def test_shrsi_overload(a: i64, b: i64) -> i64:
        return a >> b

    # CHECK:  arith.cmpi eq, %{{.*}}, %{{.*}} : i32
    def test_cmpi_eq_overload(a: i32, b: i32) -> i1:
        return a == b

    # CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : i64
    def test_cmpi_le_overload(a: i64, b: i64) -> i1:
        return a <= b

    # CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
    def test_cmpi_lt_overload(a: i32, b: i32) -> i1:
        return a < b

    # CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : i64
    def test_cmpi_ge_overload(a: i64, b: i64) -> i1:
        return a >= b

    # CHECK: arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
    def test_cmpi_gt_overload(a: i32, b: i32) -> i1:
        return a > b

    # CHECK: arith.cmpi ne, %{{.*}}, %{{.*}} : i64
    def test_cmpi_ne_overload(a: i64, b: i64) -> i1:
        return a != b

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f16
    def test_addf_overload_f16(a: f16, b: f16) -> f16:
        return a + b

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f32
    def test_addf_overload_f32(a: f32, b: f32) -> f32:
        return a + b

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f64
    def test_addf_overload_f64(a: f64, b: f64) -> f64:
        return a + b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f16
    def test_subf_overload_f16(a: f16, b: f16) -> f16:
        return a - b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f32
    def test_subf_overload_f32(a: f32, b: f32) -> f32:
        return a - b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f64
    def test_subf_overload_f64(a: f64, b: f64) -> f64:
        return a - b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f16
    def test_mulf_overload_f16(a: f16, b: f16) -> f16:
        return a * b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f32
    def test_mulf_overload_f32(a: f32, b: f32) -> f32:
        return a * b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
    def test_mulf_overload_f64(a: f64, b: f64) -> f64:
        return a * b


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Binary operation 'FloorDiv' is not supported by type '_Float64' which does not overload '__floordiv__'.
        def test_missing_floordiv_overload_f64(a: f64, b: f64) -> f64:
            # We expect the type error here, since the function doesn't exist on f64
            return a // b

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Comparison operation 'In' is not supported by type '_Float64' which does not overload '__contains__'.
        def test_missing_contains_overload_f64(a: f64, b: f64) -> f64:
            # We expect the type error here, since the function doesn't exist on f64
            return a in b

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
