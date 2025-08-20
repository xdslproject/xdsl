# RUN: python %s | filecheck %s

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram
from xdsl.frontend.pyast.utils.block import block
from xdsl.frontend.pyast.utils.const import Const
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException

p = FrontendProgram()
p.register_type(int, bigint.bigint)
p.register_type(bool, builtin.i1)


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

    def not_compiled():
        return


try:
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)


with CodeContext(p):
    # CHECK: builtin.module {
    # CHECK-NEXT: }
    pass

p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):

        def function_in_block():
            @block
            def bb1():
                # CHECK-NEXT: Cannot have a nested function 'block_inner' inside the block 'bb1'.
                def block_inner():  # pyright: ignore[reportUnusedFunction]
                    return

                return

            return bb1()

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def block_in_block():
            @block
            def bb0():
                # CHECK-NEXT: Cannot have a nested block 'bb1' inside the block 'bb0'.
                @block
                def bb1():
                    return

                return bb1()

            return bb0()

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def redefined_block():
            @block
            def bb0():  # pyright: ignore[reportRedeclaration]
                return bb0()

            # CHECK-NEXT: Block 'bb0' is already defined
            @block
            def bb0():
                return

            return bb0()

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):

        def test():
            a: Const[int] = 23  # pyright: ignore[reportAssignmentType,reportUnusedVariable]
            # CHECK-NEXT: Constant 'a' is already defined and cannot be assigned to.
            a = 3  # pyright: ignore[reportAssignmentType, reportUnusedVariable]
            return

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        a: Const[int] = 23  # pyright: ignore[reportAssignmentType]

        # CHECK-NEXT: Constant 'a' is already defined.
        def test():
            a: int = 3  # pyright: ignore[reportUnusedVariable]
            return

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        b: Const[int] = 23  # pyright: ignore[reportAssignmentType]

        def test():
            @block
            def bb0():
                # CHECK-NEXT: Constant 'b' is already defined and cannot be assigned to.
                b = 3  # pyright: ignore[reportUnusedVariable]
                return

            return bb0()

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        c: Const[int] = 23  # pyright: ignore[reportAssignmentType]

        def redefined_constant():
            # CHECK-NEXT: Constant 'c' is already defined and cannot be assigned to.
            c = 2  # pyright: ignore[reportUnusedVariable]
            return

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        c: Const[int] = 23  # pyright: ignore[reportAssignmentType]

        def foo():
            # CHECK-NEXT: Constant 'c' is already defined.
            c: int = 2  # pyright: ignore[reportUnusedVariable]
            return

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        c: Const[int] = 23  # pyright: ignore[reportAssignmentType]

        # CHECK-NEXT: Constant 'c' is already defined and cannot be used as a function/block argument name.
        def constant_as_argument(c: int):
            return

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        d: Const[int] = 23  # pyright: ignore[reportAssignmentType]

        def redefined_constant_in_block():
            @block
            def bb0():
                # CHECK-NEXT: Constant 'd' is already defined and cannot be assigned to.
                d = 2  # pyright: ignore[reportUnusedVariable]
                return

            return bb0()

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)
