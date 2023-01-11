# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    # TODO: add riscv example here.
    pass

p.compile(desymref=True)
print(p.xdsl())
