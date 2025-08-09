# RUN: python %s | filecheck %s

from collections.abc import Callable

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

p = FrontendProgram()
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
