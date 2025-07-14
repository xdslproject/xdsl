# RUN: python %s | filecheck %s

from collections.abc import Callable
from ctypes import c_int32, c_int64

from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import (
    CodeGenerationException,
    FrontendProgramException,
)
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(bool, builtin.i1)
p.register_type(c_int32, builtin.i32)
p.register_type(c_int64, builtin.i64)
try:
    with CodeContext(p):
        # CHECK: Expected non-zero number of return types in function 'test_no_return_type', but got 0.
        def test_no_return_type(a: c_int32) -> c_int32:
            return

    p.compile(desymref=False)
    exit(1)
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Type signature and the type of the return value do not match at position 0: expected i32, got i64.
        def test_wrong_return_type(a: c_int32, b: c_int64) -> c_int32:
            return b

    p.compile(desymref=False)
    exit(1)
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected no return types in function 'test_wrong_return_type'.
        def test_wrong_return_type(a: c_int32):
            return a

    p.compile(desymref=False)
    exit(1)
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected the same types for binary operation 'Add', but got i32 and i64.
        def bin_op_type_mismatch(a: c_int32, b: c_int64) -> c_int32:
            return a + b

    p.compile(desymref=False)
    exit(1)
except FrontendProgramException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected the same types for comparison operator 'Lt', but got i32 and i64.
        def cmp_op_type_mismatch(a: c_int32, b: c_int64) -> bool:
            return a < b

    p.compile(desymref=False)
    exit(1)
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
    exit(1)
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
    exit(1)
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
    exit(1)
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
    exit(1)
except CodeGenerationException as e:
    print(e.msg)


try:
    with CodeContext(p):
        # CHECK: Unsupported function argument type: 'Callable[..., None]'
        def test_complex_arg_annotation(x: Callable[..., None]) -> None:
            return

    p.compile(desymref=False)
    exit(1)
except CodeGenerationException as e:
    print(e.msg)


try:
    with CodeContext(p):
        # CHECK: Unsupported function return type: 'int | None'
        def test_complex_return_annotation() -> int | None:
            return

    p.compile(desymref=False)
    exit(1)
except CodeGenerationException as e:
    print(e.msg)
