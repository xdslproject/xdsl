from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.template import template

p = FrontendProgram()
with CodeContext(p):

    @template("N")
    def num2bits(inp: int, N: int) -> List[bool]:
        assert inp < 2 ** 64
        out = [False for i in range(N)]
        for i in range(N):
            out[i] = ((inp >> i) & 1)
        return out

    def main():
        num2bits(123, 2 ** 6)

    # def test2():
    #     num2bits(123, 32-32)

    # def test3():
    #     num2bits(123, 64)

    # def foo(x: List[int], y: List[int]) -> int:
    #     a: int = 4
    #     x[0] = 3
    #     y[0] = 4
    #     return a

    # def bar(x: List[int]):
    #     foo(x, x)
    #     x[0] = 4

p.compile(desymref=True)
# print(p.xdsl())
# print(p.mlir())

MLIR_OPT_PATH = "../llvm-project/build/bin/mlir-opt"
mlir_output = p.mlir_roundtrip(MLIR_OPT_PATH, mlir_opt_args=["--verify-each"])
print(mlir_output)
