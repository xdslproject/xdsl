# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      "func.func"() ({
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : i32):
    # CHECK-NEXT:   "symref.declare"() {"sym_name" = "x"} : () -> ()
    # CHECK-NEXT:   "symref.update"(%{{.*}}) {"symbol" = @x} : (i32) -> ()
    # CHECK-NEXT:   "func.return"() : () -> ()
    # CHECK-NEXT: }) {"sym_name" = "f1", "function_type" = (i32) -> (), "sym_visibility" = "private"} : () -> ()
    def f1(x: i32):
        pass

    # CHECK:      "func.func"() ({
    # CHECK-NEXT:   "func.return"() : () -> ()
    # CHECK-NEXT: }) {"sym_name" = "f2", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    def f2():
        return

    # CHECK:      "func.func"() ({
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : i32):
    # CHECK-NEXT:   "symref.declare"() {"sym_name" = "x"} : () -> ()
    # CHECK-NEXT:   "symref.update"(%{{.*}}) {"symbol" = @x} : (i32) -> ()
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @x} : () -> i32
    # CHECK-NEXT:   "func.return"(%{{.*}}) : (i32) -> ()
    # CHECK-NEXT: }) {"sym_name" = "f3", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()
    def f3(x: i32) -> i32:
        return x


p.compile(desymref=False)
print(p.textual_format())
