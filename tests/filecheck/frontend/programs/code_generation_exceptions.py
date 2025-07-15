# RUN: python %s | filecheck %s

from collections.abc import Callable

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

p = FrontendProgram()
try:
    with CodeContext(p):
        # CHECK: Else clause in for loops is not supported.
        def test_not_supported_loop_I():
            for _i in range(10):
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
            for _i, _j in enumerate(range(10, 0, -1)):
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
            for _i in range(0, 1, 2, 3):  # pyright: ignore[reportCallIssue]
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
            for _i, _j in range(100):  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
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
