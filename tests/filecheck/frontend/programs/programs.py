# RUN: python %s | filecheck %s

from xdsl.frontend.dialects.builtin import i32
from xdsl.frontend.const import Const
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    # CHECK: builtin.module() {}
    pass

p.compile(desymref=False)
print(p.xdsl())


p = FrontendProgram()
with CodeContext(p):

    a: Const[i32] = 1

    def foo():
        pass

p.compile(desymref=False)
print(p.xdsl())
