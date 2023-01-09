# RUN: python %s | filecheck %s

from typing import Literal, Tuple
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import TensorType, UnrankedTensorType, VectorType, i1, i32, i64, ui32, ui64, si32, si64, index, f16, f32, f64

p = FrontendProgram()
with CodeContext(p):
    #      CHECK: func.func() ["sym_name" = "bool"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !i1)
    def bool(x: i1):
        pass

    #      CHECK: func.func() ["sym_name" = "signless"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !i32, %{{.*}} : !i64)
    def signless(x: i32, y: i64):
        return

    #      CHECK: func.func() ["sym_name" = "unsigned"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !ui32, %{{.*}} : !ui64)
    def unsigned(x: ui32, y: ui64):
        return

    #      CHECK: func.func() ["sym_name" = "signed"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !si32, %{{.*}} : !si64)
    def signed(x: si32, y: si64):
        return

    #      CHECK: func.func() ["sym_name" = "indexed"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index)
    def indexed(x: index):
        return
    
    #      CHECK: func.func() ["sym_name" = "fp"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !f16, %{{.*}} : !f32, %{{.*}} : !f64)
    def fp(x: f16, y: f32, z: f64):
        return

    #      CHECK: func.func() ["sym_name" = "ranked_tensor_2"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !tensor<[2 : !index], !i32>):
    def ranked_tensor_2(x: TensorType[i32, Tuple[Literal[2],]]):
        return
    
    #      CHECK: func.func() ["sym_name" = "ranked_tensor_2_3_4"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !tensor<[2 : !index, 3 : !index, 4 : !index], !i1>):
    def ranked_tensor_2_3_4(x: TensorType[i1, Tuple[Literal[2], Literal[3], Literal[4]]]):
        return
    
    #      CHECK: func.func() ["sym_name" = "vector_4"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !vector<[4 : !index], !i32>):
    def vector_4(x: VectorType[i32, Tuple[Literal[4],]]):
        return

    #      CHECK: func.func() ["sym_name" = "unranked_tensor"
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !unranked_tensor<!i32>):
    def unranked_tensor(x: UnrankedTensorType[i32]):
        return

p.compile(desymref=False)
print(p.xdsl())
