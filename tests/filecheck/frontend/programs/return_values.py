# RUN: python %s | filecheck %s

from xdsl.dialects.builtin import i1
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

ctx = PyASTContext()
ctx.register_type(bool, i1)


@ctx.parse_program
def test_explicit_bare_return() -> None:
    return


print(test_explicit_bare_return.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_explicit_bare_return() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_early_return() -> None:
    return
    pass


try:
    test_early_return.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Return statement must be the final statement directly in the function body.


@ctx.parse_program
def test_nested_return(cond: bool) -> None:
    if cond:
        return


try:
    test_nested_return.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK: Return statement must be the final statement directly in the function body.


@ctx.parse_program
def test_multiple_returns() -> None:
    return
    return


try:
    test_multiple_returns.module
except CodeGenerationException as error:
    print(error.msg)
# CHECK-NEXT: Return statement must be the final statement directly in the function body.


@ctx.parse_program
def test_implicit_bare_return() -> None:
    pass


print(test_implicit_bare_return.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_implicit_bare_return() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_explicit_return_none() -> None:
    return None


print(test_explicit_return_none.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_explicit_return_none() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_docstring_only() -> None:
    """A docstring is skipped."""


print(test_docstring_only.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_docstring_only() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_with_docstring() -> None:
    """A docstring followed by other statements also works"""
    return None


print(test_with_docstring.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_with_docstring() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }
