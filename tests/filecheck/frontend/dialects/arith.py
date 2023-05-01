# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, i64, f16, f32, f64
from xdsl.frontend.exception import CodeGenerationException

p = FrontendProgram()
with CodeContext(p):
    # CHECK: "arith.addi"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    def test_addi_overload(a: i32, b: i32) -> i32:
        return a + b

    # CHECK: "arith.andi"(%{{.*}}, %{{.*}}) : (i64, i64) -> i64
    def test_andi_overload(a: i64, b: i64) -> i64:
        return a & b

    # CHECK: "arith.muli"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    def test_muli_overload(a: i32, b: i32) -> i32:
        return a * b

    # CHECK: "arith.subi"(%{{.*}}, %{{.*}}) : (i64, i64) -> i64
    def test_subi_overload(a: i64, b: i64) -> i64:
        return a - b

    # CHECK: "arith.shli"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    def test_shli_overload(a: i32, b: i32) -> i32:
        return a << b

    # CHECK: "arith.shrsi"(%{{.*}}, %{{.*}}) : (i64, i64) -> i64
    def test_shrsi_overload(a: i64, b: i64) -> i64:
        return a >> b

    # CHECK: "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 0 : i64} : (i32, i32) -> i1
    def test_cmpi_eq_overload(a: i32, b: i32) -> i1:
        return a == b

    # CHECK: "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 3 : i64} : (i64, i64) -> i1
    def test_cmpi_le_overload(a: i64, b: i64) -> i1:
        return a <= b

    # CHECK: "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 2 : i64} : (i32, i32) -> i1
    def test_cmpi_lt_overload(a: i32, b: i32) -> i1:
        return a < b

    # CHECK: "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 5 : i64} : (i64, i64) -> i1
    def test_cmpi_ge_overload(a: i64, b: i64) -> i1:
        return a >= b

    # CHECK: "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 4 : i64} : (i32, i32) -> i1
    def test_cmpi_gt_overload(a: i32, b: i32) -> i1:
        return a > b

    # CHECK: "arith.cmpi"(%{{.*}}, %{{.*}}) {"predicate" = 1 : i64} : (i64, i64) -> i1
    def test_cmpi_ne_overload(a: i64, b: i64) -> i1:
        return a != b

    # CHECK: "arith.addf"(%{{.*}}, %{{.*}}) : (f16, f16) -> f16
    def test_addf_overload_f16(a: f16, b: f16) -> f16:
        return a + b

    # CHECK: "arith.addf"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    def test_addf_overload_f32(a: f32, b: f32) -> f32:
        return a + b

    # CHECK: "arith.addf"(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    def test_addf_overload_f64(a: f64, b: f64) -> f64:
        return a + b

    # CHECK: "arith.subf"(%{{.*}}, %{{.*}}) : (f16, f16) -> f16
    def test_subf_overload_f16(a: f16, b: f16) -> f16:
        return a - b

    # CHECK: "arith.subf"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    def test_subf_overload_f32(a: f32, b: f32) -> f32:
        return a - b

    # CHECK: "arith.subf"(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    def test_subf_overload_f64(a: f64, b: f64) -> f64:
        return a - b

    # CHECK: "arith.mulf"(%{{.*}}, %{{.*}}) : (f16, f16) -> f16
    def test_mulf_overload_f16(a: f16, b: f16) -> f16:
        return a * b

    # CHECK: "arith.mulf"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    def test_mulf_overload_f32(a: f32, b: f32) -> f32:
        return a * b

    # CHECK: "arith.mulf"(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    def test_mulf_overload_f64(a: f64, b: f64) -> f64:
        return a * b


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Binary operation 'FloorDiv' is not supported by type '_Float64' which does not overload '__floordiv__'.
        def test_missing_floordiv_overload_f64(a: f64, b: f64) -> f64:
            # We expect the type error here, since the function doesn't exist on f64
            return a // b  # type: ignore

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Comparison operation 'In' is not supported by type '_Float64' which does not overload '__contains__'.
        def test_missing_contains_overload_f64(a: f64, b: f64) -> f64:
            # We expect the type error here, since the function doesn't exist on f64
            return a in b  # type: ignore

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
