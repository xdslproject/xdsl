// RUN: xdsl-opt -p linalg-generalize-named-ops %s | filecheck %s

//CHECK:   %lhs, %rhs, %out = "test.op"() : () -> (memref<4x5xf32>, memref<4x5xf32>, memref<4x5xf32>)
//CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (memref<2x3xf32>, memref<3x4xf32>, memref<2x4xf32>)
%lhs, %rhs, %out = "test.op"() : () -> (memref<4x5xf32>, memref<4x5xf32>, memref<4x5xf32>)
%a, %b, %c = "test.op"() : () -> (memref<2x3xf32>, memref<3x4xf32>, memref<2x4xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%lhs, %rhs : memref<4x5xf32>, memref<4x5xf32>) outs(%out : memref<4x5xf32>) {
//CHECK-NEXT:   ^bb0(%0 : f32, %1 : f32, %2 : f32):
//CHECK-NEXT:     %3 = arith.addf %0, %1 : f32
//CHECK-NEXT:     linalg.yield %3 : f32
//CHECK-NEXT:   }
linalg.add ins(%lhs, %rhs : memref<4x5xf32>, memref<4x5xf32>) outs(%out : memref<4x5xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>) outs(%c : memref<2x4xf32>) {
//CHECK-NEXT:   ^bb1(%4 : f32, %5 : f32, %6 : f32):
//CHECK-NEXT:     %7 = arith.mulf %4, %5 : f32
//CHECK-NEXT:     %8 = arith.addf %7, %6 : f32
//CHECK-NEXT:     linalg.yield %8 : f32
//CHECK-NEXT:   }
linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>) outs(%c : memref<2x4xf32>)
