# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32

p = FrontendProgram()
with CodeContext(p):
    # CHECK: builtin.module() {}
    pass

p.compile(desymref=False)
print(p.xdsl())

with CodeContext(p):
    # CHECK: builtin.module() {}
    def foo(x: i32):
        pass

p.compile(desymref=False)
print(p.xdsl())
