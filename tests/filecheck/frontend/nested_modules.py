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

p.compile()

# CHECK: builtin.module()
print(p)
