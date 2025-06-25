# RUN: python %s | filecheck %s

from ctypes import c_int64

from xdsl.dialects.builtin import i64
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
p1.register_type(c_int64, i64)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @test_identity(%a : i64) -> i64 {
    # CHECK-NEXT:     func.return %a : i64
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def test_identity(a: c_int64) -> c_int64:
        return a


p1.compile(desymref=True)
print(p1.textual_format())
