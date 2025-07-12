# RUN: python %s | filecheck %s

from xdsl.dialects.arith import AddfOp, MulfOp
from xdsl.dialects.builtin import f64
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
with CodeContext(p1):
    # CHECK: builtin.module {
    # CHECK-NEXT: }
    pass

p1.compile(desymref=False)
print(p1.textual_format())


p2 = FrontendProgram()
p2.register_type(float, f64)
p2.register_function(float.__add__, AddfOp)
p2.register_function(float.__mul__, MulfOp)
with CodeContext(p2):
    # CHECK:       builtin.module {
    # CHECK-NEXT:  func.func @foo(%x : f64, %y : f64, %z : f64) -> f64 {
    # CHECK-NEXT:    %0 = arith.mulf %y, %z : f64
    # CHECK-NEXT:    %1 = arith.addf %x, %0 : f64
    # CHECK-NEXT:    func.return %1 : f64
    # CHECK-NEXT:  }
    # CHECK-NEXT:}
    def foo(x: float, y: float, z: float) -> float:
        return x + y * z


p2.compile()
print(p2.textual_format())
