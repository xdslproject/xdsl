# RUN: python %s | filecheck %s

from xdsl.dialects.bigint import bigint
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, bigint)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%x : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:     func.return %x : !bigint.bigint
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: int) -> int:
        return x


p.compile()
print(p.textual_format())
