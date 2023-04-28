# RUN: python %s | filecheck %s

from xdsl.frontend.block import block
from xdsl.frontend.const import Const
from xdsl.frontend.exception import FrontendProgramException
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i32

p = FrontendProgram()

#      CHECK: Cannot compile program without the code context
# CHECK-NEXT:     p = FrontendProgram()
# CHECK-NEXT:     with CodeContext(p):
# CHECK-NEXT:         # Your code here.
try:
    p.compile(desymref=False)
except FrontendProgramException as e:
    print(e.msg)

#      CHECK: Cannot print the program IR without compiling it first. Make sure to use:
# CHECK-NEXT:     p = FrontendProgram()
# CHECK-NEXT:     with CodeContext(p):
# CHECK-NEXT:         # Your code here.
# CHECK-NEXT:     p.compile()
with CodeContext(p):

    def foo():
        return


try:
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():
            return

        # CHECK: Function 'foo' is already defined
        def foo():
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():
            # CHECK-NEXT: Cannot have an inner function 'bar' inside the function 'foo'.
            def bar():
                return

            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():
            @block
            def bb1():
                # CHECK-NEXT: Cannot have a nested function 'foo' inside the block 'bb1'.
                def foo():
                    return
                return

            return bb1()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():
            @block
            def bb0():
                # CHECK-NEXT: Cannot have a nested block 'bb1' inside the block 'bb0'.
                @block
                def bb1():
                    return

                return bb1()

            return bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():
            @block
            def bb0():
                return bb0()

            # CHECK-NEXT: Block 'bb0' is already defined
            @block
            def bb0():
                return

            return bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        def test():
            a: Const[i32] = 23
            # CHECK-NEXT: Constant 'a' is already defined and cannot be assigned to.
            a = 3
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        a: Const[i32] = 23
        # CHECK-NEXT: Constant 'a' is already defined.
        def test():
            a: i32 = 3
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        b: Const[i32] = 23

        def test():
            @block
            def bb0():
                # CHECK-NEXT: Constant 'b' is already defined and cannot be assigned to.
                b = 3
                return

            return bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        c: Const[i32] = 23

        def foo():
            # CHECK-NEXT: Constant 'c' is already defined and cannot be assigned to.
            c = 2
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        c: Const[i32] = 23

        def foo():
            # CHECK-NEXT: Constant 'c' is already defined.
            c: i32 = 2
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        c: Const[i32] = 23

        # CHECK-NEXT: Constant 'c' is already defined and cannot be used as a function/block argument name.
        def foo(c: i32):
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        e: Const[i32] = 23

        def foo():
            @block
            def bb0():
                # CHECK-NEXT: Constant 'e' is already defined and cannot be assigned to.
                e = 2
                return

            return bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

with CodeContext(p):
    # CHECK-NEXT: Expected non-zero number of return types in function 'foo', but got 0.
    def foo() -> i32:
        return


try:
    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)
