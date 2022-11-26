from typing import Literal, Tuple
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i64, TensorType

p = FrontendProgram()
with CodeContext(p):

    def num2bits(inp: i64, n: i64) -> TensorType[i1, Tuple[Literal[-1],]]:
        assert inp < (1 << 64)
        out: TensorType[i1, Tuple[Literal[-1],]] = [0 for i in range(n)]
        for i in range(n):
            out[i] = ((inp >> i) & 1)
        return out

p.compile()
# print(p.xdsl())
p.desymref()
# print(p.xdsl())
# print(p.mlir())

MLIR_OPT_PATH = "../llvm-project/build/bin/mlir-opt"
mlir_output = p.mlir_roundtrip(MLIR_OPT_PATH, args=["--verify-each"])
print(mlir_output)
