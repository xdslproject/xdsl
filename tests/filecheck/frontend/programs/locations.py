# RUN: python %s | filecheck %s

from xdsl.dialects import arith
from xdsl.dialects.builtin import i32
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.printer import Printer

ctx = PyASTContext(post_transforms=[])
ctx.register_type(int, i32)
ctx.register_function(int.__add__, arith.AddiOp)


@ctx.parse_program
def located(x: int) -> int:
    y = x + x
    return y


Printer(print_debuginfo=True).print_op(located.module)
print()
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @located(%x: i32 loc(unknown)) -> i32 {
# CHECK-NEXT:     symref.declare "x" loc("{{.*}}locations.py":[[# @LINE - 9]]:13)
# CHECK-NEXT:     symref.update @x = %x : i32 loc("{{.*}}locations.py":[[# @LINE - 10]]:13)
# CHECK-NEXT:     %0 = symref.fetch @x : i32 loc("{{.*}}locations.py":[[# @LINE - 10]]:13)
# CHECK-NEXT:     %1 = symref.fetch @x : i32 loc("{{.*}}locations.py":[[# @LINE - 11]]:9)
# CHECK-NEXT:     %2 = arith.addi %1, %0 : i32 loc("{{.*}}locations.py":[[# @LINE - 12]]:9)
# CHECK-NEXT:     symref.declare "y" loc("{{.*}}locations.py":[[# @LINE - 13]]:5)
# CHECK-NEXT:     symref.update @y = %2 : i32 loc("{{.*}}locations.py":[[# @LINE - 14]]:5)
# CHECK-NEXT:     %3 = symref.fetch @y : i32 loc("{{.*}}locations.py":[[# @LINE - 14]]:12)
# CHECK-NEXT:     func.return %3 : i32 loc(unknown)
# CHECK-NEXT:   } loc(unknown)
# CHECK-NEXT: } loc(unknown)


def outer() -> None:
    @ctx.parse_program
    def nested(x: int) -> int:
        return x + x

    Printer(print_debuginfo=True).print_op(nested.module)
    print()


outer()
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @nested(%x: i32 loc(unknown)) -> i32 {
# CHECK-NEXT:     symref.declare "x" loc("{{.*}}locations.py":[[# @LINE - 10]]:16)
# CHECK-NEXT:     symref.update @x = %x : i32 loc("{{.*}}locations.py":[[# @LINE - 11]]:16)
# CHECK-NEXT:     %0 = symref.fetch @x : i32 loc("{{.*}}locations.py":[[# @LINE - 11]]:20)
# CHECK-NEXT:     %1 = symref.fetch @x : i32 loc("{{.*}}locations.py":[[# @LINE - 12]]:16)
# CHECK-NEXT:     %2 = arith.addi %1, %0 : i32 loc("{{.*}}locations.py":[[# @LINE - 13]]:16)
# CHECK-NEXT:     func.return %2 : i32 loc(unknown)
# CHECK-NEXT:   } loc(unknown)
# CHECK-NEXT: } loc(unknown)
