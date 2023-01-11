# RUN: python %s | filecheck %s

from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import meta

p = FrontendProgram()
with CodeContext(p):

    #      CHECK: func.func() ["sym_name" = "num2bits_64"
    # CHECK-NEXT: ^0(%{{.*}} : !i64):
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 64 : !i64]
    # CHECK-NEXT:   %{{.*}} : !i64 = arith.constant() ["value" = 2 : !i64]
    # CHECK-NEXT:   %{{.*}} : !i64 = math.ipowi(%{{.*}} : !i64, %{{.*}} : !i64)
    # CHECK-NEXT:   %{{.*}} : !i1 = arith.cmpi(%{{.*}} : !i64, %{{.*}} : !i64) ["predicate" = 2 : !i64]
    # CHECK-NEXT:   cf.assert(%{{.*}} : !i1) ["msg" = ""]
    # CHECK-NEXT:   %{{.*}} : !i1 = arith.constant() ["value" = false]
    # CHECK-NEXT:   %{{.*}} : !tensor<[64 : !index], !i1> = tensor.splat(%{{.*}} : !i1)
    # CHECK-NEXT:   %{{.*}} : !tensor<[-1 : !index], !i1> = tensor.cast(%{{.*}} : !tensor<[64 : !index], !i1>)
    # CHECK-NEXT:   %{{.*}} : !tensor<[-1 : !index], !i1> = affine.for(%{{.*}} : !tensor<[-1 : !index], !i1>) ["lower_bound" = 0 : !index, "upper_bound" = 64 : !index, "step" = 1 : !index] {
    # CHECK-NEXT:   ^1(%{{.*}} : !index, %{{.*}} : !tensor<[-1 : !index], !i1>):
    # CHECK-NEXT:     %{{.*}} : !i64 = arith.constant() ["value" = 1 : !i64]
    # CHECK-NEXT:     %{{.*}} : !i64 = arith.index_cast(%{{.*}} : !index)
    # CHECK-NEXT:     %{{.*}} : !i64 = arith.shrsi(%{{.*}} : !i64, %{{.*}} : !i64)
    # CHECK-NEXT:     %{{.*}} : !i64 = arith.andi(%{{.*}} : !i64, %{{.*}} : !i64)
    # CHECK-NEXT:     %{{.*}} : !i1 = arith.trunci(%{{.*}} : !i64)
    # CHECK-NEXT:     %{{.*}} : !tensor<[-1 : !index], !i1> = tensor.insert(%{{.*}} : !i1, %{{.*}} : !tensor<[-1 : !index], !i1>, %{{.*}} : !index)
    # CHECK-NEXT:     affine.yield(%{{.*}} : !tensor<[-1 : !index], !i1>)
    # CHECK-NEXT:   }
    # CHECK-NEXT: func.return(%{{.*}} : !tensor<[-1 : !index], !i1>)

    #      CHECK: func.func() ["sym_name" = "main"
    # CHECK-NEXT: %{{.*}} : !i64 = arith.constant() ["value" = 123 : !i64]
    # CHECK-NEXT: %{{.*}} : !tensor<[-1 : !index], !i1> = func.call(%{{.*}} : !i64) ["callee" = @num2bits_64]
    # CHECK-NEXT: func.return()
    @meta("N")
    def num2bits(inp: int, N: int) -> List[bool]:
        assert inp < 2 ** 64
        out: List[bool] = [False for i in range(N)]
        for i in range(N):
            out[i] = ((inp >> i) & 1)
        return out

    def main():
        num2bits(123, 2 ** 6)

p.compile(desymref=True)
print(p.xdsl())
