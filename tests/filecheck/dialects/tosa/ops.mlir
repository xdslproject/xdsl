// RUN: XDSL_ROUNDTRIP

%0 = "test.op"() : () -> tensor<12x34xi32>
%1 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%2 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>


// CHECK: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> tensor<12x34xi32>
// CHECK-NEXT:   %1 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %2 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: }
