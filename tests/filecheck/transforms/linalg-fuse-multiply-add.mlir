// RUN: xdsl-opt %s -p linalg-fuse-multiply-add | filecheck %s
// RUN: xdsl-opt %s -p linalg-fuse-multiply-add{require_scalar_factor=true} | filecheck %s --check-prefix=SCALAR
// RUN: xdsl-opt %s -p linalg-fuse-multiply-add{require_erasable_mul=true} | filecheck %s --check-prefix=FOLD-MUL

builtin.module {
  %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %c = arith.constant dense<2.997925e+08> : tensor<8xf32>
  %0 = linalg.mul ins(%t0, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%t0 : tensor<8xf32>) -> tensor<8xf32>
  %1 = linalg.mul ins(%c, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) -> tensor<8xf32>
  %2 = linalg.add ins(%0, %t2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
  %3 = linalg.add ins(%1, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
  %4 = linalg.sub ins(%1, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
// CHECK-NEXT:   %c = arith.constant dense<0x4D8EF3C4> : tensor<8xf32>
// CHECK-NEXT:   %0 = linalg.mul ins(%c, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:   %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%t0, %t1, %t2 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) outs(%t0 : tensor<8xf32>) {
// CHECK-NEXT:   ^bb0(%2 : f32, %3 : f32, %4 : f32, %5 : f32):
// CHECK-NEXT:     %6 = arith.mulf %2, %3 : f32
// CHECK-NEXT:     %7 = arith.addf %6, %4 : f32
// CHECK-NEXT:     linalg.yield %7 : f32
// CHECK-NEXT:   } -> tensor<8xf32>
// CHECK-NEXT:   %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %t1, %t3 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) {
// CHECK-NEXT:   ^bb1(%9 : f32, %10 : f32, %11 : f32, %12 : f32):
// CHECK-NEXT:     %13 = arith.mulf %9, %10 : f32
// CHECK-NEXT:     %14 = arith.addf %13, %11 : f32
// CHECK-NEXT:     linalg.yield %14 : f32
// CHECK-NEXT:   } -> tensor<8xf32>
// CHECK-NEXT:   %15 = linalg.sub ins(%0, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT: }


// SCALAR-NEXT: builtin.module {
// SCALAR-NEXT:   %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
// SCALAR-NEXT:   %c = arith.constant dense<0x4D8EF3C4> : tensor<8xf32>
// SCALAR-NEXT:   %0 = linalg.mul ins(%t0, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%t0 : tensor<8xf32>) -> tensor<8xf32>
// SCALAR-NEXT:   %1 = linalg.mul ins(%c, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) -> tensor<8xf32>
// SCALAR-NEXT:   %2 = linalg.add ins(%0, %t2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// SCALAR-NEXT:   %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %t1, %t3 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) {
// SCALAR-NEXT:   ^bb0(%4 : f32, %5 : f32, %6 : f32, %7 : f32):
// SCALAR-NEXT:     %8 = arith.mulf %4, %5 : f32
// SCALAR-NEXT:     %9 = arith.addf %8, %6 : f32
// SCALAR-NEXT:     linalg.yield %9 : f32
// SCALAR-NEXT:   } -> tensor<8xf32>
// SCALAR-NEXT:   %10 = linalg.sub ins(%1, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
// SCALAR-NEXT: }


// FOLD-MUL-NEXT: builtin.module {
// FOLD-MUL-NEXT:   %t0, %t1, %t2, %t3 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
// FOLD-MUL-NEXT:   %c = arith.constant dense<0x4D8EF3C4> : tensor<8xf32>
// FOLD-MUL-NEXT:   %0 = linalg.mul ins(%c, %t1 : tensor<8xf32>, tensor<8xf32>) outs(%c : tensor<8xf32>) -> tensor<8xf32>
// FOLD-MUL-NEXT:   %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%t0, %t1, %t2 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) outs(%t0 : tensor<8xf32>) {
// FOLD-MUL-NEXT:   ^bb0(%2 : f32, %3 : f32, %4 : f32, %5 : f32):
// FOLD-MUL-NEXT:     %6 = arith.mulf %2, %3 : f32
// FOLD-MUL-NEXT:     %7 = arith.addf %6, %4 : f32
// FOLD-MUL-NEXT:     linalg.yield %7 : f32
// FOLD-MUL-NEXT:   } -> tensor<8xf32>
// FOLD-MUL-NEXT:   %8 = linalg.add ins(%0, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// FOLD-MUL-NEXT:   %9 = linalg.sub ins(%0, %t3 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
// FOLD-MUL-NEXT: }
