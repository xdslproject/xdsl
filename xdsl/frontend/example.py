from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

def template(*params):
    def decorate(f):
        pass
    return decorate

p = FrontendProgram()
with CodeContext(p):

    @template("n")
    def num2bits(inp: int, n: int) -> List[bool]:
        assert inp < 2 ** 64
        out = [False for i in range(n)]
        n = n + 1
        for i in range(n):
            out[i] = ((inp >> i) & 1) + n
        return out

    def test1():
        num2bits(123, 128)
    
    def test2():
        num2bits(123, 2 ** 5)

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
