# RUN: python %s | filecheck %s

from ctypes import c_float, c_int32, c_int64

from xdsl.dialects import arith, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException


def add_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def and_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def mul_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def sub_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def shl_i32(operand1: c_int32, operand2: c_int32) -> c_int32: ...
def shrs_i64(operand1: c_int64, operand2: c_int64) -> c_int64: ...
def cmp_eq_i32(operand1: c_int32, operand2: c_int32) -> bool: ...
def cmp_le_i64(operand1: c_int64, operand2: c_int64) -> bool: ...
def cmp_lt_i32(operand1: c_int32, operand2: c_int32) -> bool: ...
def cmp_ge_i64(operand1: c_int64, operand2: c_int64) -> bool: ...
def cmp_gt_i32(operand1: c_int32, operand2: c_int32) -> bool: ...
def cmp_ne_i64(operand1: c_int64, operand2: c_int64) -> bool: ...
def add_f32(operand1: c_float, operand2: c_float) -> c_float: ...
def sub_f32(operand1: c_float, operand2: c_float) -> c_float: ...
def mul_f32(operand1: c_float, operand2: c_float) -> c_float: ...


ctx = PyASTContext(post_transforms=[])
ctx.register_type(c_int32, builtin.i32)
ctx.register_type(c_int64, builtin.i64)
ctx.register_type(c_float, builtin.f32)
ctx.register_type(float, builtin.f64)
ctx.register_type(bool, builtin.i1)
ctx.register_function(add_i32, arith.AddiOp)
ctx.register_function(and_i64, arith.AndIOp)
ctx.register_function(mul_i32, arith.MuliOp)
ctx.register_function(sub_i64, arith.SubiOp)
ctx.register_function(shl_i32, arith.ShLIOp)
ctx.register_function(shrs_i64, arith.ShRSIOp)
ctx.register_function(cmp_eq_i32, lambda x, y: arith.CmpiOp(x, y, "eq"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
ctx.register_function(cmp_le_i64, lambda x, y: arith.CmpiOp(x, y, "sle"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
ctx.register_function(cmp_lt_i32, lambda x, y: arith.CmpiOp(x, y, "slt"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
ctx.register_function(cmp_ge_i64, lambda x, y: arith.CmpiOp(x, y, "sge"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
ctx.register_function(cmp_gt_i32, lambda x, y: arith.CmpiOp(x, y, "sgt"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
ctx.register_function(cmp_ne_i64, lambda x, y: arith.CmpiOp(x, y, "ne"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
ctx.register_function(add_f32, arith.AddfOp)
ctx.register_function(float.__add__, arith.AddfOp)
ctx.register_function(sub_f32, arith.SubfOp)
ctx.register_function(float.__sub__, arith.SubfOp)
ctx.register_function(mul_f32, arith.MulfOp)
ctx.register_function(float.__mul__, arith.MulfOp)


# CHECK: arith.addi %{{.*}}, %{{.*}} : i32
@ctx.parse_program
def test_addi_overload(a: c_int32, b: c_int32) -> c_int32:
    return add_i32(a, b)


print(test_addi_overload.module)


# CHECK: arith.andi %{{.*}}, %{{.*}} : i64
@ctx.parse_program
def test_andi_overload(a: c_int64, b: c_int64) -> c_int64:
    return and_i64(a, b)


print(test_andi_overload.module)


# CHECK: arith.muli %{{.*}}, %{{.*}} : i32
@ctx.parse_program
def test_muli_overload(a: c_int32, b: c_int32) -> c_int32:
    return mul_i32(a, b)


print(test_muli_overload.module)


# CHECK: arith.subi %{{.*}}, %{{.*}} : i64
@ctx.parse_program
def test_subi_overload(a: c_int64, b: c_int64) -> c_int64:
    return sub_i64(a, b)


print(test_subi_overload.module)


# CHECK: arith.shli %{{.*}}, %{{.*}} : i32
@ctx.parse_program
def test_shli_overload(a: c_int32, b: c_int32) -> c_int32:
    return shl_i32(a, b)


print(test_shli_overload.module)


# CHECK: arith.shrsi %{{.*}}, %{{.*}} : i64
@ctx.parse_program
def test_shrsi_overload(a: c_int64, b: c_int64) -> c_int64:
    return shrs_i64(a, b)


print(test_shrsi_overload.module)


# CHECK:  arith.cmpi eq, %{{.*}}, %{{.*}} : i32
@ctx.parse_program
def test_cmpi_eq_overload(a: c_int32, b: c_int32) -> bool:
    return cmp_eq_i32(a, b)


print(test_cmpi_eq_overload.module)


# CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : i64
@ctx.parse_program
def test_cmpi_le_overload(a: c_int64, b: c_int64) -> bool:
    return cmp_le_i64(a, b)


print(test_cmpi_le_overload.module)


# CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
@ctx.parse_program
def test_cmpi_lt_overload(a: c_int32, b: c_int32) -> bool:
    return cmp_lt_i32(a, b)


print(test_cmpi_lt_overload.module)


# CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : i64
@ctx.parse_program
def test_cmpi_ge_overload(a: c_int64, b: c_int64) -> bool:
    return cmp_ge_i64(a, b)


print(test_cmpi_ge_overload.module)


# CHECK: arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
@ctx.parse_program
def test_cmpi_gt_overload(a: c_int32, b: c_int32) -> bool:
    return cmp_gt_i32(a, b)


print(test_cmpi_gt_overload.module)


# CHECK: arith.cmpi ne, %{{.*}}, %{{.*}} : i64
@ctx.parse_program
def test_cmpi_ne_overload(a: c_int64, b: c_int64) -> bool:
    return cmp_ne_i64(a, b)


print(test_cmpi_ne_overload.module)


# CHECK: arith.addf %{{.*}}, %{{.*}} : f32
@ctx.parse_program
def test_addf_overload_f32(a: c_float, b: c_float) -> c_float:
    return add_f32(a, b)


print(test_addf_overload_f32.module)


# CHECK: arith.addf %{{.*}}, %{{.*}} : f64
@ctx.parse_program
def test_addf_overload_f64(a: float, b: float) -> float:
    return a + b


print(test_addf_overload_f64.module)


# CHECK: arith.subf %{{.*}}, %{{.*}} : f32
@ctx.parse_program
def test_subf_overload_f32(a: c_float, b: c_float) -> c_float:
    return sub_f32(a, b)


print(test_subf_overload_f32.module)


# CHECK: arith.subf %{{.*}}, %{{.*}} : f64
@ctx.parse_program
def test_subf_overload_f64(a: float, b: float) -> float:
    return a - b


print(test_subf_overload_f64.module)


# CHECK: arith.mulf %{{.*}}, %{{.*}} : f32
@ctx.parse_program
def test_mulf_overload_f32(a: c_float, b: c_float) -> c_float:
    return mul_f32(a, b)


print(test_mulf_overload_f32.module)


# CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
@ctx.parse_program
def test_mulf_overload_f64(a: float, b: float) -> float:
    return a * b


print(test_mulf_overload_f64.module)


# CHECK: Binary operation 'FloorDiv' is not supported by type 'float' which does not overload '__floordiv__'.
@ctx.parse_program
def test_missing_floordiv_overload_f64(a: float, b: float) -> float:
    # We expect the type error here, since the function doesn't exist on f64
    return a // b


try:
    test_missing_floordiv_overload_f64.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK: Comparison operation 'In' is not supported by type 'f64' which does not overload '__contains__'.
@ctx.parse_program
def test_missing_contains_overload_f64(a: float, b: float) -> float:
    # We expect the type error here, since the function doesn't exist on f64
    return a in b  # pyright: ignore[reportOperatorIssue]


try:
    test_missing_contains_overload_f64.module
except CodeGenerationException as e:
    print(e.msg)
