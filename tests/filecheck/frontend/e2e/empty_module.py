# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    # CHECK: builtin.module() {}
    pass

p.compile()
print(p.xdsl())
