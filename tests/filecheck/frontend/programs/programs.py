# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK: builtin.module {
    # CHECK-NEXT: }
    pass

p.compile(desymref=False)
print(p.textual_format())
