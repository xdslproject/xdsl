from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import meta

p = FrontendProgram()
with CodeContext(p):

    @meta("N")
    def num2bits(inp: int, N: int) -> List[bool]:
        assert inp < 2 ** 64
        out = [False for i in range(N)]
        for i in range(N):
            out[i] = ((inp >> i) & 1)
        return out

    def main():
        num2bits(123, 2 ** 6)

p.compile(desymref=True)
# print(p.xdsl())
# print(p.mlir())

MLIR_OPT_PATH = "../llvm-project/build/bin/mlir-opt"
mlir_output = p.mlir_roundtrip(MLIR_OPT_PATH, mlir_opt_args=["--verify-each"])
print(mlir_output)
