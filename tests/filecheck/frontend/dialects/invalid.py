# RUN: python %s | filecheck %s

from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, i64
from xdsl.frontend.exception import CodeGenerationException, FrontendProgramException
from xdsl.frontend.program import FrontendProgram

p = FrontendProgram()

try:
    with CodeContext(p):
        # CHECK: Expected non-zero number of return types in function 'test_no_return_type', but got 0.
        def test_no_return_type(a: i32) -> i32:
            return

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Type signature and the type of the return value do not match at position 0: expected i32, got i64.
        def test_wrong_return_type(a: i32, b: i64) -> i32:
            return b

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected no return types in function 'test_wrong_return_type'.
        def test_wrong_return_type(a: i32):
            return a

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected the same types for binary operation 'Add', but got i32 and i64.
        def bin_op_type_mismatch(a: i32, b: i64) -> i32:
            return a + b

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected the same types for comparison operator 'Lt', but got i32 and i64.
        def cmp_op_type_mismatch(a: i32, b: i64) -> i1:
            return a < b

    p.compile(desymref=False)
    print(p.textual_format())
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Else clause in for loops is not supported.
        def test_not_supported_loop_I():
            for i in range(10):
                pass
            else:
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Only range-based loops are supported.
        def test_not_supported_loop_II():
            for i, j in enumerate(range(10, 0, -1)):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Range-based loop expected between 1 and 3 arguments, but got 4.
        def test_not_supported_loop_III():
            for i in range(0, 1, 2, 3):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Range-based loop must take a single target variable, e.g. `for i in range(10)`.
        def test_not_supported_loop_IV():
            for i, j in range(100):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
