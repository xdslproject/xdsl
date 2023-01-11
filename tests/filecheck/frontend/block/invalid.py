# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import block
from tests.filecheck.frontend.utils import assert_excepton


p = FrontendProgram()


with CodeContext(p):
    
    # CHECK: Found a block 'bad_block' which does not belong to any function. All blocks have to be inside functions.
    @block
    def bad_block():
        x = 3

assert_excepton(p)


with CodeContext(p):
    
    # CHECK: Block 'bb0' in function 'foo' cannot return anything.
    def foo():
        @block
        def bb0() -> None:
            x = 3

assert_excepton(p)


with CodeContext(p):
    
    def foo():
        # CHECK: Unresolved symbol function or block called 'bb0'.
        bb0()

        @block
        def bb1():
            return

assert_excepton(p)

with CodeContext(p):
    
    def foo():
        a: int = 12
        # CHECK: Block 'bb1' expected 2 arguments, but got 1.
        bb1(a)

        @block
        def bb1(x: int, y: int):
            return

assert_excepton(p, desymref=False)
