from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):

    def num2bits(inp: int, n: int) -> List[bool]:
        assert inp < 2 ** 64
        out = [False for i in range(n)]
        for i in range(n):
            out[i] = ((inp >> i) & 1)
        return out

p.compile()
# print(p.xdsl())
# print(p.mlir())

MLIR_OPT_PATH = "../llvm-project/build/bin/mlir-opt"
mlir_output = p.mlir_roundtrip(MLIR_OPT_PATH, args=["--verify-each"])
print(mlir_output)
