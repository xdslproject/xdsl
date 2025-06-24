# RUN: python %s | filecheck %s

from xdsl.dialects.bigint import AddOp, LtOp, MulOp, SubOp, bigint
from xdsl.dialects.builtin import i1
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, bigint)
with CodeContext(p):
    # CHECK:      builtin.module {
    # CHECK-NEXT:   func.func @foo(%x : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:     func.return %x : !bigint.bigint
    # CHECK-NEXT:   }
    # CHECK-NEXT: }
    def foo(x: int) -> int:
        return x


p.compile()
print(p.textual_format())


p1 = FrontendProgram()
p1.register_type(int, bigint)
p1.register_function(int.__add__, AddOp)
p1.register_function(int.__sub__, SubOp)
p1.register_function(int.__mul__, MulOp)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:     func.func @bar(%x : !bigint.bigint, %y : !bigint.bigint, %z : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:       %0 = bigint.mul %y, %z : !bigint.bigint
    # CHECK-NEXT:       %1 = bigint.sub %x, %0 : !bigint.bigint
    # CHECK-NEXT:       func.return %1 : !bigint.bigint
    # CHECK-NEXT:     }
    # CHECK-NEXT:   }
    def bar(x: int, y: int, z: int) -> int:
        return x - y * z


p1.compile()
print(p1.textual_format())


p2 = FrontendProgram()
p2.register_type(int, bigint)
p2.register_type(bool, i1)
p2.register_function(int.__lt__, LtOp)
with CodeContext(p2):
    # CHECK:      builtin.module {
    # CHECK-NEXT:       func.func @zee(%x : !bigint.bigint, %y : !bigint.bigint) -> i1 {
    # CHECK-NEXT:         %0 = bigint.lt %x, %y : i1
    # CHECK-NEXT:         func.return %0 : i1
    # CHECK-NEXT:       }
    # CHECK-NEXT:     }
    def zee(x: int, y: int) -> bool:
        return x < y


p2.compile()
print(p2.textual_format())
