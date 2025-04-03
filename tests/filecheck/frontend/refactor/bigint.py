# RUN: python %s | filecheck %s

from xdsl.dialects.bigint import BigIntegerType
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, BigIntegerType)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%0 : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:     func.return %0 : !bigint.bigint
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: int) -> int:
        return x


p.compile()
print(p.textual_format())
