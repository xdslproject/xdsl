# RUN: python %s | filecheck %s

from typing import Literal, Tuple
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import index, i64, f32, TensorType
from xdsl.frontend.dialects import tensor
from xdsl.frontend.dialects.tensor import insert

p = FrontendProgram()
with CodeContext(p):

    # CHECK: %{{.*}} : !i64 = tensor.extract(%{{.*}} : !tensor<[2 : !index, 4 : !index], !i64>, %{{.*}} : !index, %{{.*}} : !index)
    def test_extract(x: TensorType[i64, Tuple[Literal[2], Literal[4],]], i: index, j: index) -> i64:
        return tensor.extract(x, i, j)
    
    # CHECK: %{{.*}} : !f32 = tensor.extract(%{{.*}} : !tensor<[4 : !index, 3 : !index, 2 : !index], !f32>, %{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index)
    def test_extract_overload(x: TensorType[f32, Tuple[Literal[4], Literal[3], Literal[2],]], i: index, j: index, k: index) -> f32:
        return x[i][j][k]


    # CHECK: tensor.insert(%{{.*}} : !i64, %{{.*}} : !tensor<[2 : !index], !i64>, %{{.*}} : !index)
    def test_insert(v: i64, x: TensorType[i64, Tuple[Literal[2],]], i: index):
        insert(v, x, i)
    
    # CHECK: tensor.insert(%{{.*}} : !i64, %{{.*}} : !tensor<[2 : !index, 3 : !index], !i64>, %{{.*}} : !index, %{{.*}} : !index)
    def test_insert_overload(v: i64, x: TensorType[i64, Tuple[Literal[2], Literal[3],]], i: index):
        j: index = 0
        x[i][j] = v

p.compile()
print(p.xdsl())

