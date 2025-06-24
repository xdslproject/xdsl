# RUN: python %s | filecheck %s

from xdsl.dialects.arith import AddfOp, AddiOp, MulfOp, SubfOp
from xdsl.dialects.builtin import I64, IntegerAttr, f64, i64
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
p1.register_type(IntegerAttr[I64], i64)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @test_identity(%a : i64) -> i64 {
    # CHECK-NEXT:     func.return %a : i64
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def test_identity(a: IntegerAttr[I64]) -> IntegerAttr[I64]:
        return a


p1.compile(desymref=True)
print(p1.textual_format())


p2 = FrontendProgram()
p2.register_type(IntegerAttr[I64], i64)
p2.register_function(IntegerAttr.__add__, AddiOp)
with CodeContext(p2):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @test_addi(%a : i64, %b : i64) -> i64 {
    # CHECK-NEXT:     %0 = arith.addi %a, %b : i64
    # CHECK-NEXT:     func.return %0 : i64
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def test_addi(a: IntegerAttr[I64], b: IntegerAttr[I64]) -> IntegerAttr[I64]:
        return a + b


p2.compile(desymref=True)
print(p2.textual_format())

p3 = FrontendProgram()
p3.register_type(float, f64)
p3.register_function(float.__add__, AddfOp)
p3.register_function(float.__sub__, SubfOp)
p3.register_function(float.__mul__, MulfOp)
with CodeContext(p3):
    # CHECK:      builtin.module {
    # CHECK-NEXT:       func.func @bar(%x : f64, %y : f64, %z : f64) -> f64 {
    # CHECK-NEXT:         %0 = arith.mulf %y, %z : f64
    # CHECK-NEXT:         %1 = arith.subf %x, %0 : f64
    # CHECK-NEXT:         func.return %1 : f64
    # CHECK-NEXT:       }
    # CHECK-NEXT:     }
    def bar(x: float, y: float, z: float) -> float:
        return x - y * z


p3.compile()
print(p3.textual_format())
