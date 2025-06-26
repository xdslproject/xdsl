# RUN: python %s | filecheck %s

from ctypes import c_float, c_int32, c_int64

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram


def add_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def and_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def mul_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def sub_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def shl_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def shrs_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def cmp_eq_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def cmp_le_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def cmp_lt_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def cmp_ge_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def cmp_gt_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def cmp_ne_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def add_f32(operand1: c_float, operand2: c_float) -> c_float: ...
def sub_f32(operand1: c_float, operand2: c_float) -> c_float: ...
def mul_f32(operand1: c_float, operand2: c_float) -> c_float: ...


p = FrontendProgram()
p.register_type(c_int32, builtin.i32)
p.register_type(c_int64, builtin.i64)
p.register_type(c_float, builtin.f32)
p.register_type(float, builtin.f64)
p.register_type(bool, builtin.i1)
p.register_function(add_i32, arith.AddiOp)
p.register_function(and_i64, arith.AndIOp)
p.register_function(mul_i32, arith.MuliOp)
p.register_function(sub_i64, arith.SubiOp)
p.register_function(shl_i32, arith.ShLIOp)
p.register_function(shrs_i64, arith.ShRSIOp)
p.register_function(cmp_eq_i32, lambda x, y: arith.CmpiOp(x, y, "eq"))
p.register_function(cmp_le_i64, lambda x, y: arith.CmpiOp(x, y, "sle"))
p.register_function(cmp_lt_i32, lambda x, y: arith.CmpiOp(x, y, "slt"))
p.register_function(cmp_ge_i64, lambda x, y: arith.CmpiOp(x, y, "sge"))
p.register_function(cmp_gt_i32, lambda x, y: arith.CmpiOp(x, y, "sgt"))
p.register_function(cmp_ne_i64, lambda x, y: arith.CmpiOp(x, y, "ne"))
p.register_function(add_f32, arith.AddfOp)
p.register_function(float.__add__, arith.AddfOp)
p.register_function(sub_f32, arith.SubfOp)
p.register_function(float.__sub__, arith.SubfOp)
p.register_function(mul_f32, arith.MulfOp)
p.register_function(float.__mul__, arith.MulfOp)
with CodeContext(p):
    # CHECK: arith.addi %{{.*}}, %{{.*}} : i32
    def test_addi_overload(a: c_int32, b: c_int32) -> c_int32:
        return add_i32(a, b)

    # CHECK: arith.andi %{{.*}}, %{{.*}} : i64
    def test_andi_overload(a: c_int64, b: c_int64) -> c_int64:
        return and_i64(a, b)

    # CHECK: arith.muli %{{.*}}, %{{.*}} : i32
    def test_muli_overload(a: c_int32, b: c_int32) -> c_int32:
        return mul_i32(a, b)

    # CHECK: arith.subi %{{.*}}, %{{.*}} : i64
    def test_subi_overload(a: c_int64, b: c_int64) -> c_int64:
        return sub_i64(a, b)

    # CHECK: arith.shli %{{.*}}, %{{.*}} : i32
    def test_shli_overload(a: c_int32, b: c_int32) -> c_int32:
        return shl_i32(a, b)

    # CHECK: arith.shrsi %{{.*}}, %{{.*}} : i64
    def test_shrsi_overload(a: c_int64, b: c_int64) -> c_int64:
        return shrs_i64(a, b)

    # CHECK:  arith.cmpi eq, %{{.*}}, %{{.*}} : i32
    def test_cmpi_eq_overload(a: c_int32, b: c_int32) -> bool:
        return cmp_eq_i32(a, b)

    # CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : i64
    def test_cmpi_le_overload(a: c_int64, b: c_int64) -> bool:
        return cmp_le_i64(a, b)

    # CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
    def test_cmpi_lt_overload(a: c_int32, b: c_int32) -> bool:
        return cmp_lt_i32(a, b)

    # CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : i64
    def test_cmpi_ge_overload(a: c_int64, b: c_int64) -> bool:
        return cmp_ge_i64(a, b)

    # CHECK: arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
    def test_cmpi_gt_overload(a: c_int32, b: c_int32) -> bool:
        return cmp_gt_i32(a, b)

    # CHECK: arith.cmpi ne, %{{.*}}, %{{.*}} : i64
    def test_cmpi_ne_overload(a: c_int64, b: c_int64) -> bool:
        return cmp_ne_i64(a, b)

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f32
    def test_addf_overload_f32(a: c_float, b: c_float) -> c_float:
        return add_f32(a, b)

    # CHECK: arith.addf %{{.*}}, %{{.*}} : f64
    def test_addf_overload_f64(a: float, b: float) -> float:
        return a + b

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f32
    def test_subf_overload_f32(a: c_float, b: c_float) -> c_float:
        return sub_f32(a, b)

    # CHECK: arith.subf %{{.*}}, %{{.*}} : f64
    def test_subf_overload_f64(a: float, b: float) -> float:
        return a - b

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f32
    def test_mulf_overload_f32(a: c_float, b: c_float) -> c_float:
        return mul_f32(a, b)

    # CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
    def test_mulf_overload_f64(a: float, b: float) -> float:
        return a * b


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Binary operation 'FloorDiv' is not supported by type 'float' which does not overload '__floordiv__'.
        def test_missing_floordiv_overload_f64(a: float, b: float) -> float:
            # We expect the type error here, since the function doesn't exist on f64
            return a // b

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Comparison operation 'In' is not supported by type 'f64' which does not overload '__contains__'.
        def test_missing_contains_overload_f64(a: float, b: float) -> float:
            # We expect the type error here, since the function doesn't exist on f64
            return a in b

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
