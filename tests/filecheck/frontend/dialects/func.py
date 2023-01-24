# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32

p = FrontendProgram()
with CodeContext(p):
    #      CHECK: func.func() ["sym_name" = "f1", "function_type" = !fun<[!i32], []>
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !i32):
    # CHECK-NEXT:   symref.declare() ["sym_name" = "{{.*}}"]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i32) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   func.return()
    def f1(x: i32):
        pass

    #      CHECK: func.func() ["sym_name" = "f2", "function_type" = !fun<[], []>
    # CHECK-NEXT:   func.return()
    def f2():
        return
    
    #      CHECK: func.func() ["sym_name" = "f3", "function_type" = !fun<[!i32], [!i32]>
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !i32):
    # CHECK-NEXT:   symref.declare() ["sym_name" = "{{.*}}"]
    # CHECK-NEXT:   symref.update(%{{.*}} : !i32) ["symbol" = @{{.*}}]
    # CHECK-NEXT:   %{{.*}} : !i32 = symref.fetch() ["symbol" = @{{.*}}]
    # CHECK-NEXT:   func.return(%{{.*}} : !i32)
    def f3(x: i32) -> i32:
        return x

p.compile(desymref=False)
print(p.xdsl())
