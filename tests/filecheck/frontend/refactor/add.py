# RUN: python %s | filecheck %s

from xdsl.dialects.bigint import AddBigIntOp, BigIntegerType
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, BigIntegerType)
p.register_method(int, "__add__", AddBigIntOp)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%0 : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:     %1 = bigint.add %0, %0 : !bigint.bigint
    # CHECK-NEXT:     func.return %1 : !bigint.bigint
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return x + x


p.compile()
print(p.textual_format())
