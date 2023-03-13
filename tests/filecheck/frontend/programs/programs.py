# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    # CHECK: builtin.module() {
    # CHECK-NEXT: }
    pass

p.compile(desymref=False)
print(p.xdsl())
