// RUN: XDSL_ROUNDTRIP

%0 = "test.op"() : () -> tensor<12x34xi32>
%1 = "test.op"() : () -> tensor<1x1xi32>
%2 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%3 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%4 = tosa.add %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%5 = tosa.add %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
%6 = tosa.sub %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%7 = tosa.sub %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
%8 = tosa.mul %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%9 = tosa.mul %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>


// CHECK: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> tensor<12x34xi32>
// CHECK-NEXT:   %1 = "test.op"() : () -> tensor<1x1xi32>
// CHECK-NEXT:   %2 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %3 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %4 = tosa.add %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %5 = tosa.add %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %6 = tosa.sub %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %7 = tosa.sub %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %8 = tosa.mul %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %9 = tosa.mul %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: }
