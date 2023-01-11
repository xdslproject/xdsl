# RUN: python %s | filecheck %s

from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import meta
from tests.filecheck.frontend.utils import assert_excepton


p = FrontendProgram()


def test_meta(*params):
    def decorate(f):
        pass
    return decorate


with CodeContext(p):

    # CHECK: Function {{.*}} has unknown decorator. For decorating the function as a template, use '@meta(..)'.
    @test_meta()
    def test():
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Function {{.*}} has 2 decorators but can only have 1 to mark it as a template.
    @test_meta()
    @meta("N")
    def test(N: int):
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Template for function {{.*}} must have at least one template argument.
    @meta()
    def test():
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Template for function {{.*}} has 2 template arguments, but function expects only 1 argument.
    @meta("N", "M")
    def test(N: int):
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Template for function {{.*}} has unused template arguments. All template arguments must be named exactly the same as the corresponding function arguments.
    @meta("A")
    def test(N: int):
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Cannot assign to template parameter 'N' in function {{.*}}.
    @meta("N")
    def test(N: bool):
        N = True

    def main():
        test(3)

assert_excepton(p)


with CodeContext(p):

    # CHECK: Cannot redefine the template parameter 'N' in function {{.*}}.
    @meta("N")
    def test(N: int):
        N: int = 3

    def main():
        test(3)

assert_excepton(p)


# TODO: This test should be removed when non-primitive template arguments are supported.
with CodeContext(p):

    # CHECK: Call to function {{.*}} has non-primitive template argument of type 'list' at position 0. Only primitive type arguments like int or float are supported at the moment.
    @meta("L")
    def test(L: List[int]):
        pass

    def main():
        test([0, 1, 2])

assert_excepton(p)


with CodeContext(p):

    # CHECK: Division by zero in template instantiation for function {{.*}}.
    @meta("N")
    def test(N: int) -> int:
        return N + 3

    def main():
        a: int = test(34 / 0)

assert_excepton(p)


with CodeContext(p):

    @meta("X")
    def bar(X: int) -> int:
        return X

    # CHECK: Non-template argument 'x' in template instantiation for function 'bar'.
    @meta("A", "B")
    def foo(A: int, x: int, B: int) -> int:
        return A - x + bar(x)
    
    def main():
        a: int = foo(1, 2, 3)

assert_excepton(p)


with CodeContext(p):

    @meta("X")
    def bar(X: int) -> int:
        return X

    # CHECK: Non-template argument 'constant' in template instantiation for function 'bar'.
    def foo() -> int:
        constant: int = 45
        return bar(constant)
    
    def main():
        a: int = foo()

assert_excepton(p)
