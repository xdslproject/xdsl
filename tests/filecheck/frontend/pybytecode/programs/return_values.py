# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import PyASTContext

ctx = PyASTContext()


@ctx.parse_program
def test_explicit_bare_return():
    return


print(test_explicit_bare_return.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_explicit_bare_return() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_explicit_bare_return_with_annotation() -> None:
    return


print(test_explicit_bare_return_with_annotation.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_explicit_bare_return_with_annotation() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_implicit_bare_return():
    pass


print(test_implicit_bare_return.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_implicit_bare_return() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_implicit_bare_return_with_annotation() -> None:
    pass


print(test_implicit_bare_return_with_annotation.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_implicit_bare_return_with_annotation() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_explicit_return_none():
    return None


print(test_explicit_return_none.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_explicit_return_none() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def test_explicit_return_none_with_annotation():
    return None


print(test_explicit_return_none_with_annotation.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @test_explicit_return_none_with_annotation() {
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }
