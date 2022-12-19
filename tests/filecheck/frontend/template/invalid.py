# RUN: python %s | filecheck %s

from typing import List
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.template import template


p = FrontendProgram()


def assert_excepton(p: FrontendProgram):
    try:
        p.compile()
        assert False, "should not compile"
    except CodegenException as e:
        print(e.msg)


def test_template(*params):
    def decorate(f):
        pass
    return decorate


with CodeContext(p):

    # CHECK: Function {{.*}} has unknown decorator. For decorating the function as a template, use '@template(..)'.
    @test_template()
    def test():
        pass

assert_excepton(p)


p = FrontendProgram()
with CodeContext(p):

    # CHECK: Template for function {{.*}} must have at least one template argument.
    @template()
    def test():
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Function {{.*}} has 2 decorators but can only have 1 to mark it as a template.
    @test_template()
    @template("N")
    def test(N: int):
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Template for function {{.*}} must have at least one template argument.
    @template()
    def test():
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Template for function {{.*}} has 2 template arguments, but function expects only 1 argument.
    @template("N", "M")
    def test(N: int):
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Template for function {{.*}} has unused template arguments. All template arguments must be named exactly the same as the corresponding function arguments.
    @template("A")
    def test(N: int):
        pass

assert_excepton(p)


with CodeContext(p):

    # CHECK: Cannot assign to template parameter 'N' in function {{.*}}.
    @template("N")
    def test(N: bool):
        N = True

    def main():
        test(3)

assert_excepton(p)

with CodeContext(p):

    # CHECK: Cannot redefine the template parameter 'N' in function {{.*}}.
    @template("N")
    def test(N: int):
        N: int = 3

    def main():
        test(3)

assert_excepton(p)

# TODO: This test should be removed when non-primitive template arguments are supported.
with CodeContext(p):

    # CHECK: Call to function {{.*}} has non-primitive template argument of type 'list' at position 0. Only primitive type arguments like int or float are supported at the moment.
    @template("L")
    def test(L: List[int]):
        pass

    def main():
        test([0, 1, 2])

assert_excepton(p)


with CodeContext(p):

    # CHECK: Invalid template instantiation for function {{.*}}; ZeroDivisionError: division by zero.
    @template("N")
    def test(N: int) -> int:
        return N + 3

    def main():
        a: int = test(34 / 0)

assert_excepton(p)
