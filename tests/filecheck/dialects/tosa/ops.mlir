// RUN: XDSL_ROUNDTRIP

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

// -----
// CHECK-LABEL: clamp
func.func @test_clamp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %{{.*}} = tosa.clamp %{{.*}} {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.clamp %arg0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: concat
func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  // CHECK: %{{.*}} = tosa.concat %{{.*}}, %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----
// CHECK-LABEL: cos
func.func @test_cos(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %{{.*}} = tosa.cos %{{.*}} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.cos %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: matmul
// TODO: use zp operands for MLIR v21
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  // CHECK: %0 = tosa.matmul %{{.*}}, %{{.*}} : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  %0 = tosa.matmul %arg0, %arg1 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
// CHECK-LABEL: test_mul
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>, %shift: tensor<1xi8>) -> tensor<13x21x3xf32> {
  // CHECK: %{{.*}} = tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xf32>, tensor<13x1x3xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xf32>, tensor<13x1x3xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: mul_i32
func.func @test_mul_i32(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>, %shift: tensor<1xi8>) -> tensor<13x21x3xi32> {
  // CHECK: %{{.*}} = tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xi32>, tensor<13x1x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi32>, tensor<13x1x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}
