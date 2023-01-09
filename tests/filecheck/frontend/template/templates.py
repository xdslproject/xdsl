# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.template import template


p = FrontendProgram()
with CodeContext(p):

    @template("A", "B")
    def foo(x: int, A: int, B: int) -> int:
        return A - x + B
    
    # CHECK: func.func() ["sym_name" = "foo_1_3", "function_type" = !fun<[!i64], [!i64]>
    # CHECK: func.func() ["sym_name" = "foo_1_4", "function_type" = !fun<[!i64], [!i64]>
    
    # CHECK: %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @foo_1_3]
    # CHECK: %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @foo_1_3]
    # CHECK: %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @foo_1_4]
    def simple_template():
        a: int = foo(2, 1, 3)
        b: int = foo(1, 1, 3)
        b: int = foo(2, 1, 4)

p.compile()
print(p.xdsl())


with CodeContext(p):

    # CHECK: func.func() ["sym_name" = "baz_2_3", "function_type" = !fun<[!i64], [!i64]>
    # CHECK:  %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @bar_2_12]
    # CHECK:  %{{.*}} : !i64 = func.call() ["callee" = @foo_3]

    # CHECK: func.func() ["sym_name" = "foo_3", "function_type" = !fun<[], [!i64]>

    # CHECK: func.func() ["sym_name" = "bar_2_12", "function_type" = !fun<[!i64], [!i64]>
    # CHECK: %{{.*}} : !i64 = func.call() ["callee" = @foo_14]

    # CHECK: func.func() ["sym_name" = "foo_14", "function_type" = !fun<[], [!i64]>

    @template("X")
    def foo(X: int) -> int:
        return X

    @template("A", "B")
    def bar(x: int, A: int, B: int) -> int:
        return A - x + foo(A+B)
    
    @template("A", "B")
    def baz(x: int, A: int, B: int) -> int:
        return x - foo(B) + bar(x, A, 12)
    
    def nested_templates():
        x: int = baz(1, 2, 3)

p.compile()
print(p.xdsl())


with CodeContext(p):

    @template("N")
    def factorial(N: int) -> int:
        result: int = 1 if N == 1 else N * factorial(N if N <= 1 else N - 1)
        return result

    # CHECK: func.func() ["sym_name" = "factorial_5", "function_type" = !fun<[], [!i64]>
    # CHECK: %{{.*}} : !i64 = func.call() ["callee" = @factorial_4]

    # CHECK: func.func() ["sym_name" = "factorial_4", "function_type" = !fun<[], [!i64]>
    # CHECK: %{{.*}} : !i64 = func.call() ["callee" = @factorial_3]

    # CHECK: func.func() ["sym_name" = "factorial_3", "function_type" = !fun<[], [!i64]>
    # CHECK: %{{.*}} : !i64 = func.call() ["callee" = @factorial_2]

    # CHECK: func.func() ["sym_name" = "factorial_2", "function_type" = !fun<[], [!i64]>
    # CHECK: %{{.*}} : !i64 = func.call() ["callee" = @factorial_1]

    # CHECK: func.func() ["sym_name" = "factorial_1", "function_type" = !fun<[], [!i64]>
    def main():
        answer: int = factorial(5)

p.compile()
print(p.xdsl())
