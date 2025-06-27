# RUN: python %s | filecheck %s

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
p1.register_type(int, bigint.bigint)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%x : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:     func.return %x : !bigint.bigint
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: int) -> int:
        return x


p1.compile()
print(p1.textual_format())


p2 = FrontendProgram()
p2.register_type(int, bigint.bigint)
p2.register_type(float, builtin.f64)
p2.register_type(bool, builtin.i1)
p2.register_function(int.__add__, bigint.AddOp)
p2.register_function(int.__sub__, bigint.SubOp)
p2.register_function(int.__mul__, bigint.MulOp)
p2.register_function(int.__floordiv__, bigint.FloorDivOp)
p2.register_function(int.__mod__, bigint.ModOp)
p2.register_function(int.__pow__, bigint.PowOp)
p2.register_function(int.__lshift__, bigint.LShiftOp)
p2.register_function(int.__rshift__, bigint.RShiftOp)
p2.register_function(int.__or__, bigint.BitOrOp)
p2.register_function(int.__xor__, bigint.BitXorOp)
p2.register_function(int.__and__, bigint.BitAndOp)
p2.register_function(int.__truediv__, bigint.DivOp)
p2.register_function(int.__eq__, bigint.EqOp)
p2.register_function(int.__ne__, bigint.NeqOp)
p2.register_function(int.__gt__, bigint.GtOp)
p2.register_function(int.__ge__, bigint.GteOp)
p2.register_function(int.__lt__, bigint.LtOp)
p2.register_function(int.__le__, bigint.LteOp)
with CodeContext(p2):
    # CHECK: bigint.add %{{.*}}, %{{.*}} : !bigint.bigint
    def test_add_overload(a: int, b: int) -> int:
        return a + b

    # CHECK: bigint.sub %{{.*}}, %{{.*}} : !bigint.bigint
    def test_sub_overload(a: int, b: int) -> int:
        return a - b

    # CHECK: bigint.mul %{{.*}}, %{{.*}} : !bigint.bigint
    def test_mul_overload(a: int, b: int) -> int:
        return a * b

    # CHECK: bigint.floordiv %{{.*}}, %{{.*}} : !bigint.bigint
    def test_floordiv_overload(a: int, b: int) -> int:
        return a // b

    # CHECK: bigint.mod %{{.*}}, %{{.*}} : !bigint.bigint
    def test_mod_overload(a: int, b: int) -> int:
        return a % b

    # CHECK: bigint.pow %{{.*}}, %{{.*}} : !bigint.bigint
    def test_pow_overload(a: int, b: int) -> int:
        return a**b

    # CHECK: bigint.lshift %{{.*}}, %{{.*}} : !bigint.bigint
    def test_lshift_overload(a: int, b: int) -> int:
        return a << b

    # CHECK: bigint.rshift %{{.*}}, %{{.*}} : !bigint.bigint
    def test_rshift_overload(a: int, b: int) -> int:
        return a >> b

    # CHECK: bigint.bitor %{{.*}}, %{{.*}} : !bigint.bigint
    def test_or_overload(a: int, b: int) -> int:
        return a | b

    # CHECK: bigint.bitxor %{{.*}}, %{{.*}} : !bigint.bigint
    def test_xor_overload(a: int, b: int) -> int:
        return a ^ b

    # CHECK: bigint.bitand %{{.*}}, %{{.*}} : !bigint.bigint
    def test_and_overload(a: int, b: int) -> int:
        return a & b

    # CHECK: bigint.div %{{.*}}, %{{.*}} : f64
    def test_div_overload(a: int, b: int) -> float:
        return a / b

    # CHECK: bigint.eq %{{.*}}, %{{.*}} : i1
    def test_eq_overload(a: int, b: int) -> bool:
        return a == b

    # CHECK: bigint.neq %{{.*}}, %{{.*}} : i1
    def test_neq_overload(a: int, b: int) -> bool:
        return a != b

    # CHECK: bigint.gt %{{.*}}, %{{.*}} : i1
    def test_gt_overload(a: int, b: int) -> bool:
        return a > b

    # CHECK: bigint.gte %{{.*}}, %{{.*}} : i1
    def test_gte_overload(a: int, b: int) -> bool:
        return a >= b

    # CHECK: bigint.lt %{{.*}}, %{{.*}} : i1
    def test_lt_overload(a: int, b: int) -> bool:
        return a < b

    # CHECK: bigint.lte %{{.*}}, %{{.*}} : i1
    def test_lte_overload(a: int, b: int) -> bool:
        return a <= b


p2.compile()
print(p2.textual_format())
