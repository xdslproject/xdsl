# RUN: python %s | filecheck %s

from xdsl.dialects.bigint import AddBigIntOp, BigIntegerType, MulBigIntOp, SubBigIntOp
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, BigIntegerType)
p.register_method(int, "__add__", AddBigIntOp)
p.register_method(int, "__sub__", SubBigIntOp)
p.register_method(int, "__mul__", MulBigIntOp)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:     func.func @foo(%0 : !bigint.bigint, %1 : !bigint.bigint, %2 : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:       %3 = bigint.mul %1, %2 : !bigint.bigint
    # CHECK-NEXT:       %4 = bigint.sub %0, %3 : !bigint.bigint
    # CHECK-NEXT:       func.return %4 : !bigint.bigint
    # CHECK-NEXT:     }
    # CHECK-NEXT:   }
    def foo(x: int, y: int, z: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return x - y * z


p.compile()
print(p.textual_format())
