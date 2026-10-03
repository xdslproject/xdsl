# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import PyASTContext

ctx = PyASTContext()


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
