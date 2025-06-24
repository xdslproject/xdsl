# RUN: python %s | filecheck %s

from xdsl.dialects.arith import AddfOp, MulfOp, SubfOp
from xdsl.dialects.builtin import f64
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p2 = FrontendProgram()
p2.register_type(float, f64)
p2.register_function(float.__add__, AddfOp)
p2.register_function(float.__sub__, SubfOp)
p2.register_function(float.__mul__, MulfOp)
with CodeContext(p2):
    # CHECK:      builtin.module {
    # CHECK-NEXT:       func.func @bar(%x : f64, %y : f64, %z : f64) -> f64 {
    # CHECK-NEXT:         %0 = arith.mulf %y, %z : f64
    # CHECK-NEXT:         %1 = arith.subf %x, %0 : f64
    # CHECK-NEXT:         func.return %1 : f64
    # CHECK-NEXT:       }
    # CHECK-NEXT:     }
    def bar(x: float, y: float, z: float) -> float:
        return x - y * z


p2.compile()
print(p2.textual_format())
