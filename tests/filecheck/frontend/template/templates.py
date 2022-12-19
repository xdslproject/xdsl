# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.template import template


p = FrontendProgram()
with CodeContext(p):

    @template("A", "B")
    def foo(A: int, x: int, B: int) -> int:
        return A - x + B
    
    # CHECK: func.func() ["sym_name" = "foo_1_3", "function_type" = !fun<[!i64], [!i64]>
    # CHECK: func.func() ["sym_name" = "foo_1_4", "function_type" = !fun<[!i64], [!i64]>
    
    def main():
        # CHECK: %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @foo_1_3]
        a: int = foo(1, 2, 3)
        # CHECK: %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @foo_1_3]
        b: int = foo(1, 1, 3)
        # CHECK: %{{.*}} : !i64 = func.call(%{{.*}} : !i64) ["callee" = @foo_1_4]
        b: int = foo(1, 2, 4)


    # TODO: Enable this test when we support evaluation of expressions which use template arguments.
    # @template("X")
    # def bar(X: int) -> int:
    #     return X

    # @template("A", "B")
    # def foo(A: int, x: int, B: int) -> int:
    #     return A - x + bar(A+B)
    
    # def main():
    #     a: int = foo(1, 2, 3)


p.compile()
print(p.xdsl())
