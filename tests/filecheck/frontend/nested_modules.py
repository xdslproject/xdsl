# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import Module

p = FrontendProgram()
with CodeContext(p):
    with Module():
        with Module():
            pass
    with Module():
        with Module():
            with Module():
                pass
    with Module():
        pass

#      CHECK: builtin.module() {
# CHECK-NEXT:   builtin.module() {
# CHECK-NEXT:     builtin.module() {}
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module() {
# CHECK-NEXT:     builtin.module() {
# CHECK-NEXT:       builtin.module() {}
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module() {}
# CHECK-NEXT: }
p.compile()
print(p)
