# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32

p = FrontendProgram()
with CodeContext(p):
    #      CHECK: func.func() ["sym_name" = "foo", "function_type" = !fun<[!i32], []>
    # CHECK-NEXT: ^0(%{{.*}} : !i32):
    # CHECK-NEXT:   symref.declare() ["sym_name" = "{{.*}}"]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i32) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   func.return()
    def foo(x: i32):
        pass

p.compile(desymref=False)
print(p.xdsl())
