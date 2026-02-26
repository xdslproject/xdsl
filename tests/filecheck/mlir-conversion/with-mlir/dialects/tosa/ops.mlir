// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// CHECK-LABEL: const
func.func @const() -> () {
  %out = "tosa.const"() <{values = dense<0> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  // CHECK: %0 = "tosa.const"() <{values = dense<0> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  return
}

// -----
// CHECK-LABEL: avg_pool2d_f32
func.func @test_avg_pool2d_f32(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
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
  %0 = tosa.clamp %arg0 {min_val = 0 : i32, max_val = 1 : i32} : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
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
// CHECK-LABEL: test_matmul
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
// CHECK-LABEL: test_mul
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xf32>, tensor<13x1x3xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: i32_mul
func.func @test_i32_mul(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
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
func.func @test_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
    %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
    %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
    %input_zp = "tosa.const"() <{values = dense<127> : tensor<1xi8>}> : () -> tensor<1xi8>
    %output_zp = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "SINGLE_ROUND", scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
    return %0 : tensor<13x21x3xi8>
}

// -----
// CHECK-LABEL: test_sin
func.func @test_sin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %{{.*}} = tosa.sin %{{.*}} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.sin %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: recip_i32
func.func @test_recip_i32(%arg0: tensor<12x24xi32>) -> tensor<12x24xi32> {
  // CHECK: {{%.*}} = tosa.reciprocal {{%.*}} : (tensor<12x24xi32>) -> tensor<12x24xi32>
  %0 = tosa.reciprocal %arg0 : (tensor<12x24xi32>) -> tensor<12x24xi32>
  return %0 : tensor<12x24xi32>
}

// -----
// CHECK-LABEL: recip_f32
func.func @test_recip_f32(%arg0: tensor<12x24xf32>) -> tensor<12x24xf32> {
  // CHECK: {{%.*}} = tosa.reciprocal {{%.*}} : (tensor<12x24xf32>) -> tensor<12x24xf32>
  %0 = tosa.reciprocal %arg0 : (tensor<12x24xf32>) -> tensor<12x24xf32>
  return %0 : tensor<12x24xf32>
}

// -----
// CHECK-LABEL: reduce_all
func.func @test_reduce_all(%arg0: tensor<31x5x3xi1>) -> tensor<1x5x3xi1> {
  // CHECK: {{%.*}} = tosa.reduce_all %{{.*}} {axis = 0 : i32} : (tensor<31x5x3xi1>) -> tensor<1x5x3xi1>
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<31x5x3xi1>) -> tensor<1x5x3xi1>
  return %0 : tensor<1x5x3xi1>
}

// -----
// CHECK-LABEL: reduce_any
func.func @test_reduce_any(%arg0: tensor<31x5x3xi1>) -> tensor<1x5x3xi1> {
  // CHECK: {{%.*}} = tosa.reduce_any %{{.*}} {axis = 0 : i32} : (tensor<31x5x3xi1>) -> tensor<1x5x3xi1>
  %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<31x5x3xi1>) -> tensor<1x5x3xi1>
  return %0 : tensor<1x5x3xi1>
}

// -----
// CHECK-LABEL: reduce_max
func.func @test_reduce_max(%arg0: tensor<31x5x3xf32>) -> tensor<1x5x3xf32> {
  // CHECK: {{%.*}} = tosa.reduce_max {{%.*}} {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  return %0 : tensor<1x5x3xf32>
}

// -----
// CHECK-LABEL: reduce_min
func.func @test_reduce_min(%arg0: tensor<31x5x3xf32>) -> tensor<1x5x3xf32> {
  // CHECK: {{%.*}} = tosa.reduce_min {{%.*}} {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  return %0 : tensor<1x5x3xf32>
}

// -----
// CHECK-LABEL: reduce_product
func.func @test_reduce_product(%arg0: tensor<31x5x3xf32>) -> tensor<1x5x3xf32> {
  // CHECK: {{%.*}} = tosa.reduce_product {{%.*}} {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  return %0 : tensor<1x5x3xf32>
}

// -----
// CHECK-LABEL: reduce_sum
func.func @test_reduce_sum(%arg0: tensor<31x5x3xf32>) -> tensor<1x5x3xf32> {
  // CHECK: {{%.*}} = tosa.reduce_sum {{%.*}} {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<31x5x3xf32>) -> tensor<1x5x3xf32>
  return %0 : tensor<1x5x3xf32>
}

// -----
// CHECK-LABEL: cond_if
func.func @test_cond_if(%arg0: tensor<i1>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  // CHECK: {{%.*}} = tosa.cond_if {{%.*}} -> (tensor<3x4x5xf32>) {
  %0 = tosa.cond_if %arg0 -> (tensor<3x4x5xf32>) {
    // CHECK-NEXT: tosa.yield {{%.*}} : tensor<3x4x5xf32>
    tosa.yield %arg1 : tensor<3x4x5xf32>
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: tosa.yield {{%.*}} : tensor<3x4x5xf32>
    tosa.yield %arg1 : tensor<3x4x5xf32>
    // CHECK-NEXT: }
  }
  return %0 : tensor<3x4x5xf32>
}
