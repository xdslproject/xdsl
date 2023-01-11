# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import block
from tests.filecheck.frontend.utils import assert_excepton

p = FrontendProgram()
with CodeContext(p):
    
    # CHECK: Found a block 'bb0' which does not belong to any function. All blocks have to be inside functions.
    @block
    def bb0():
        x = 3

assert_excepton(p)
