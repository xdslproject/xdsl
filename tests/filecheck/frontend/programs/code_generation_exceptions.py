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
def test_for_unsupported(end: c_size_t) -> None:
    for _ in range(
        end  # pyright: ignore[reportArgumentType]
    ):
        pass
    return


try:
    test_for_unsupported.module
except NotImplementedError as e:
    print(e)


# CHECK: Code generation exception at {{.*}}code_generation_exceptions.py:[[# @LINE + 2]]:33: Unsupported function argument type: 'Callable[..., None]'
@ctx.parse_program
def test_complex_arg_annotation(x: Callable[..., None]) -> None:
    return


try:
    test_complex_arg_annotation.module
except CodeGenerationException as e:
    print(e)


# CHECK: Code generation exception at {{.*}}code_generation_exceptions.py:[[# @LINE + 2]]:1: Unsupported function return type: 'int | None'
@ctx.parse_program
def test_complex_return_annotation() -> int | None:
    return


try:
    test_complex_return_annotation.module
except CodeGenerationException as e:
    print(e)


def test_nested_location():
    # CHECK: Code generation exception at {{.*}}code_generation_exceptions.py:[[# @LINE + 2]]:30: Unsupported function argument type: 'float'
    @ctx.parse_program
    def nested(é: c_size_t, x: float) -> None:
        return

    try:
        nested.module
    except CodeGenerationException as e:
        print(e)


test_nested_location()
