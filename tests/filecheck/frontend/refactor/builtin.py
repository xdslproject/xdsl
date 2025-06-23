# RUN: python %s | filecheck %s

from xdsl.dialects import arith
from xdsl.dialects.builtin import I64, IntegerAttr, i64
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(IntegerAttr[I64], i64)
p.register_function(IntegerAttr.__add__, arith.AddiOp)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @test_addi_overload(%a : i64, %b : i64) -> i64 {
    # CHECK-NEXT:     %0 = arith.addi %a, %b : i64
    # CHECK-NEXT:     func.return %0 : i64
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def test_addi_overload(
        a: IntegerAttr[I64], b: IntegerAttr[I64]
    ) -> IntegerAttr[I64]:
        return a + b


p.compile(desymref=True)
print(p.textual_format())
