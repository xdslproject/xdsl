# RUN: python "%s" | filecheck "%s"

from xdsl.frontend.context import CodeContext
from xdsl.frontend.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK: builtin.module {
    # CHECK-NEXT: }
    pass

p.compile(desymref=False)
print(p.textual_format())
