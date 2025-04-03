# RUN: python %s | filecheck %s

from xdsl.dialects.arith import AddfOp, MulfOp, SubfOp
from xdsl.dialects.bigint import AddOp, BigIntegerType, MulOp, SubOp
from xdsl.dialects.builtin import Float64Type
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
p1.register_type(int, BigIntegerType)
p1.register_method(int, "__add__", AddOp)
p1.register_method(int, "__sub__", SubOp)
p1.register_method(int, "__mul__", MulOp)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:     func.func @foo(%0 : !bigint.bigint, %1 : !bigint.bigint, %2 : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:       %3 = bigint.mul %1, %2 : !bigint.bigint
    # CHECK-NEXT:       %4 = bigint.sub %0, %3 : !bigint.bigint
    # CHECK-NEXT:       func.return %4 : !bigint.bigint
    # CHECK-NEXT:     }
    # CHECK-NEXT:   }
    def foo(x: int, y: int, z: int) -> int:
        return x - y * z


p1.compile()
print(p1.textual_format())


p2 = FrontendProgram()
p2.register_type(float, Float64Type)
p2.register_method(float, "__add__", AddfOp)
p2.register_method(float, "__sub__", SubfOp)
p2.register_method(float, "__mul__", MulfOp)
with CodeContext(p2):
    # CHECK:      builtin.module {
    # CHECK-NEXT:       func.func @bar(%0 : f64, %1 : f64, %2 : f64) -> f64 {
    # CHECK-NEXT:         %3 = arith.mulf %1, %2 : f64
    # CHECK-NEXT:         %4 = arith.subf %0, %3 : f64
    # CHECK-NEXT:         func.return %4 : f64
    # CHECK-NEXT:       }
    # CHECK-NEXT:     }
    def bar(x: float, y: float, z: float) -> float:
        return x - y * z


p2.compile()
print(p2.textual_format())
