# RUN: python %s | filecheck %s

from xdsl.dialects import arith
from xdsl.dialects.builtin import I64, IntegerAttr, i64
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
p1.register_type(IntegerAttr[I64], i64, annotation_name="IntegerAttr[I64]")
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
p2.register_type(IntegerAttr[I64], i64, annotation_name="IntegerAttr[I64]")
p2.register_function(IntegerAttr.__add__, arith.AddiOp)
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
