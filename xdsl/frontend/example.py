from typing import Literal, Tuple
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i64, TensorType

p = FrontendProgram()
with CodeContext(p):

    def num2bits(inp: i64) -> TensorType[i1, Tuple[Literal[64],]]:
        assert inp < (1 << 64)
        out: TensorType[i1, Tuple[Literal[64],]] = [0 for i in range(64)]
        for i in range(64):
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
