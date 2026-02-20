# RUN: python %s | filecheck %s

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import PyASTContext

ctx1 = PyASTContext()
ctx1.register_type(int, bigint.bigint)


# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @foo(%x : !bigint.bigint) -> !bigint.bigint {
# CHECK-NEXT:     func.return %x : !bigint.bigint
# CHECK-NEXT:   }
# CHECK-NEXT: }
@ctx1.parse_program
def foo(x: int) -> int:
    return x


print(foo.module)


# CHECK: bigint.constant 0
@ctx1.parse_program
def test_constant_int_zero() -> int:
    return 0


print(test_constant_int_zero.module)


# CHECK: bigint.constant 42
@ctx1.parse_program
def test_constant_int_nonzero() -> int:
    return 42


print(test_constant_int_nonzero.module)


# CHECK: bigint.constant 123456789012345678901234567890
@ctx1.parse_program
def test_constant_int_large() -> int:
    return 123456789012345678901234567890


print(test_constant_int_large.module)


ctx2 = PyASTContext()
ctx2.register_type(int, bigint.bigint)
ctx2.register_type(float, builtin.f64)
ctx2.register_type(bool, builtin.i1)
ctx2.register_function(int.__add__, bigint.AddOp)
ctx2.register_function(int.__sub__, bigint.SubOp)
ctx2.register_function(int.__mul__, bigint.MulOp)
ctx2.register_function(int.__floordiv__, bigint.FloorDivOp)
ctx2.register_function(int.__mod__, bigint.ModOp)
ctx2.register_function(int.__pow__, bigint.PowOp)
ctx2.register_function(int.__lshift__, bigint.LShiftOp)
ctx2.register_function(int.__rshift__, bigint.RShiftOp)
ctx2.register_function(int.__or__, bigint.BitOrOp)
ctx2.register_function(int.__xor__, bigint.BitXorOp)
ctx2.register_function(int.__and__, bigint.BitAndOp)
ctx2.register_function(int.__truediv__, bigint.DivOp)
ctx2.register_function(int.__eq__, bigint.EqOp)
ctx2.register_function(int.__ne__, bigint.NeqOp)
ctx2.register_function(int.__gt__, bigint.GtOp)
ctx2.register_function(int.__ge__, bigint.GteOp)
ctx2.register_function(int.__lt__, bigint.LtOp)
ctx2.register_function(int.__le__, bigint.LteOp)


# CHECK: bigint.add %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_add_overload(a: int, b: int) -> int:
    return a + b


print(test_add_overload.module)


# CHECK: bigint.sub %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_sub_overload(a: int, b: int) -> int:
    return a - b


print(test_sub_overload.module)


# CHECK: bigint.mul %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_mul_overload(a: int, b: int) -> int:
    return a * b


print(test_mul_overload.module)


# CHECK: bigint.floordiv %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_floordiv_overload(a: int, b: int) -> int:
    return a // b


print(test_floordiv_overload.module)


# CHECK: bigint.mod %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_mod_overload(a: int, b: int) -> int:
    return a % b


print(test_mod_overload.module)


# CHECK: bigint.pow %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_pow_overload(a: int, b: int) -> int:
    return a**b


print(test_pow_overload.module)


# CHECK: bigint.lshift %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_lshift_overload(a: int, b: int) -> int:
    return a << b


print(test_lshift_overload.module)


# CHECK: bigint.rshift %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_rshift_overload(a: int, b: int) -> int:
    return a >> b


print(test_rshift_overload.module)


# CHECK: bigint.bitor %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_or_overload(a: int, b: int) -> int:
    return a | b


print(test_or_overload.module)


# CHECK: bigint.bitxor %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_xor_overload(a: int, b: int) -> int:
    return a ^ b


print(test_xor_overload.module)


# CHECK: bigint.bitand %{{.*}}, %{{.*}} : !bigint.bigint
@ctx2.parse_program
def test_and_overload(a: int, b: int) -> int:
    return a & b


print(test_and_overload.module)


# CHECK: bigint.div %{{.*}}, %{{.*}} : f64
@ctx2.parse_program
def test_div_overload(a: int, b: int) -> float:
    return a / b


print(test_div_overload.module)


# CHECK: bigint.eq %{{.*}}, %{{.*}} : i1
@ctx2.parse_program
def test_eq_overload(a: int, b: int) -> bool:
    return a == b


print(test_eq_overload.module)


# CHECK: bigint.neq %{{.*}}, %{{.*}} : i1
@ctx2.parse_program
def test_neq_overload(a: int, b: int) -> bool:
    return a != b


print(test_neq_overload.module)


# CHECK: bigint.gt %{{.*}}, %{{.*}} : i1
@ctx2.parse_program
def test_gt_overload(a: int, b: int) -> bool:
    return a > b


print(test_gt_overload.module)


# CHECK: bigint.gte %{{.*}}, %{{.*}} : i1
@ctx2.parse_program
def test_gte_overload(a: int, b: int) -> bool:
    return a >= b


print(test_gte_overload.module)


# CHECK: bigint.lt %{{.*}}, %{{.*}} : i1
@ctx2.parse_program
def test_lt_overload(a: int, b: int) -> bool:
    return a < b


print(test_lt_overload.module)


# CHECK: bigint.lte %{{.*}}, %{{.*}} : i1
@ctx2.parse_program
def test_lte_overload(a: int, b: int) -> bool:
    return a <= b


print(test_lte_overload.module)
