// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// CHECK-LABEL: avg_pool2d_f32
func.func @test_avg_pool2d_f32(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
  // CHECK: %{{.*}} = tosa.avg_pool2d %{{.*}} {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32>
  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----
// CHECK-LABEL: binop
func.func @test_binop(%arg0: tensor<12x34xi32>, %arg1: tensor<1x1xi32>) -> (tensor<12x34xi32>, tensor<12x34xi32>, tensor<12x34xi32>, tensor<12x34xi32>) {
// CHECK: %{{.*}} = tosa.add %{{.*}}, %{{.*}} : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
  %add = tosa.add %arg0, %arg0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK: %{{.*}} = tosa.add %{{.*}}, %{{.*}} : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
  %add_broadcast = tosa.add %arg0, %arg1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK: %{{.*}} = tosa.sub %{{.*}}, %{{.*}} : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
  %sub = tosa.sub %arg0, %arg0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK: %{{.*}} = tosa.sub %{{.*}}, %{{.*}} : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
  %sub_broadcast = tosa.sub %arg0, %arg1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
  return %add, %add_broadcast, %sub, %sub_broadcast : tensor<12x34xi32>, tensor<12x34xi32>, tensor<12x34xi32>, tensor<12x34xi32>
}
