# RUN: python %s | filecheck %s

from xdsl.frontend.block import block
from xdsl.frontend.const import Const
from xdsl.frontend.exception import FrontendProgramException
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, i64

p = FrontendProgram()

try:
    with CodeContext(p):
        # CHECK: Expected non-zero number of return types in function 'test_no_return_type', but got 0.
        def test_no_return_type(a: i32) -> i32:
            return

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Type signature and the type of the return value do not match at position 0: expected i32, got i64.
        def test_wrong_return_type(a: i32, b: i64) -> i32:
            return b

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected no return types in function 'test_wrong_return_type'.
        def test_wrong_return_type(a: i32):
            return a

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected the same types for binary operation 'Add', but got i32 and i64.
        def bin_op_type_mismatch(a: i32, b: i64) -> i32:
            return a + b

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected the same types for comparison operator 'Lt', but got i32 and i64.
        def cmp_op_type_mismatch(a: i32, b: i64) -> i1:
            return a < b

    p.compile(desymref=False)
    print(p.xdsl())
except FrontendProgramException as e:
    print(e.msg)
