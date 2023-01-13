# RUN: python %s | filecheck %s

from xdsl.frontend.block import block
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
            pass

        # CHECK: Function 'foo' is already defined.
        def foo():
            pass

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():

            # CHECK: Cannot have an inner function 'bar' inside the function 'foo'.
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
                # CHECK: Cannot have an inner function 'foo' inside the block 'bb1'.
                def foo():
                    return

            bb1()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():

            @block
            def bb0():

                # CHECK: Cannot have a nested block 'bb1' inside the block 'bb0'.
                @block
                def bb1():
                    return

                bb1()

            bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def foo():

            @block
            def bb0():
                bb0()

            # Block 'bb0' is already defined in function 'foo'.
            @block
            def bb0():
                return

            bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        @block
        def bb0():
            bb0()

        # CHECK: Block 'bb0' is already defined.
        @block
        def bb0():
            return

        bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        @block
        def bb0():
            # CHECK: Cannot have an inner function 'foo' inside the block 'bb0'.
            def foo():
                return

        bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        @block
        def bb0():
            # CHECK: Cannot have a nested block 'bb1' inside the block 'bb0'.
            @block
            def bb1():
                return

            bb1()

        bb0()

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

with CodeContext(p):
    # CHECK: Expected 'foo' to return a type.
    def foo() -> i32:
        pass


try:
    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)
