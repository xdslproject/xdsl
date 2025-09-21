// RUN: XDSL_ROUNDTRIP

// CHECK-LABEL: avg_pool2d_f32
func.func @test_avg_pool2d_f32(%arg0: tensor<1x7x7x9xf32>, %input_zp: tensor<1xf32>, %output_zp: tensor<1xf32>) -> tensor<1x7x7x9xf32> {
  // CHECK: %{{.*}} = tosa.avg_pool2d %{{.*}}, %{{.*}}, %{{.*}} {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
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
// CHECK-LABEL: clamp_f32
func.func @test_clamp_f32(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %{{.*}} = tosa.clamp %{{.*}} {max_val = 1.000000e+00 : f32, min_val = 0.000000e+00 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.clamp %arg0 {max_val = 1.000000e+00 : f32, min_val = 0.000000e+00 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: clamp_i32
func.func @test_clamp_i32(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: %{{.*}} = tosa.clamp %{{.*}} {max_val = 1 : i32, min_val = 0 : i32} : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = tosa.clamp %arg0 {max_val = 1 : i32, min_val = 0 : i32} : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
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
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>, %azp0: tensor<1xf32>, %bzp0: tensor<1xf32>) -> tensor<1x14x28xf32> {
  // CHECK: %0 = tosa.matmul %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x14x28xf32>
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
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

// -----
// CHECK-LABEL: max_pool2d_f32
func.func @test_max_pool2d_f32(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // CHECK: %{{.*}} = tosa.max_pool2d %{{.*}} {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: max_pool2d_bf16
func.func @test_max_pool2d_bf16(%arg0: tensor<1x32x32x8xbf16>) -> tensor<1x32x32x8xbf16> {
  // CHECK: %{{.*}} = tosa.max_pool2d %{{.*}} {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xbf16>) -> tensor<1x32x32x8xbf16>
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xbf16>) -> tensor<1x32x32x8xbf16>
  return %0 : tensor<1x32x32x8xbf16>
}

// -----
// CHECK-LABEL: max_pool2d_f16
func.func @test_max_pool2d_f16(%arg0: tensor<1x32x32x8xf16>) -> tensor<1x32x32x8xf16> {
  // CHECK: %{{.*}} = tosa.max_pool2d %{{.*}} {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf16>) -> tensor<1x32x32x8xf16>
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf16>) -> tensor<1x32x32x8xf16>
  return %0 : tensor<1x32x32x8xf16>
}

// -----
// CHECK-LABEL: rescale
func.func @test_rescale(%arg0: tensor<13x21x3xi32>, %multiplier: tensor<1xi32>, %shift: tensor<1xi8>, %input_zp: tensor<1xi32>, %output_zp: tensor<1xi32>) -> tensor<13x21x3xi32> {
    // CHECK: %{{.*}} = tosa.rescale %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
    %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
    return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: test_sin
func.func @test_sin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %{{.*}} = tosa.sin %{{.*}} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.sin %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
