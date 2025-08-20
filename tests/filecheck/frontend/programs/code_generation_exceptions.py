# RUN: python %s | filecheck %s

from collections.abc import Callable
from ctypes import c_size_t

from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

ctx = PyASTContext(post_transforms=[])
ctx.register_type(c_size_t, builtin.IndexType())


# CHECK: For loops are currently not supported!
@ctx.parse_program
def test_for_unsupported(end: c_size_t):
    for _ in range(
        end  # pyright: ignore[reportArgumentType]
    ):
        pass
    return


try:
    test_for_unsupported.module
except NotImplementedError as e:
    print(e)


# CHECK: Unsupported function argument type: 'Callable[..., None]'
@ctx.parse_program
def test_complex_arg_annotation(x: Callable[..., None]) -> None:
    return


try:
    test_complex_arg_annotation.module
except CodeGenerationException as e:
    print(e.msg)


# CHECK: Unsupported function return type: 'int | None'
@ctx.parse_program
def test_complex_return_annotation() -> int | None:
    return


try:
    test_complex_return_annotation.module
except CodeGenerationException as e:
    print(e.msg)
