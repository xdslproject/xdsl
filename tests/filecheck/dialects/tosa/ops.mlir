// RUN: XDSL_ROUNDTRIP

%0 = "test.op"() : () -> tensor<12x34xi32>
%1 = "test.op"() : () -> tensor<1x1xi32>
%2 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%3 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%4 = tosa.add %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%5 = tosa.add %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
%6 = tosa.sub %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%7 = tosa.sub %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>

%f = "test.op"() : () -> tensor<12x13xf32>
%s = "test.op"() : () -> tensor<1xi8>
%8 = tosa.mul %f, %f, %s : (tensor<12x13xf32>, tensor<12x13xf32>, tensor<1xi8>) -> tensor<12x13xf32> 
%9 = tosa.mul %f, %f : (tensor<12x13xf32>, tensor<12x13xf32>) -> tensor<12x13xf32> 
%10 = tosa.mul %0, %1, %s : (tensor<12x34xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<12x34xi32>

%11 = tosa.sin %f : (tensor<12x13xf32>) -> tensor<12x13xf32>
%12 = tosa.cos %f : (tensor<12x13xf32>) -> tensor<12x13xf32>

%m = "test.op"() : () -> tensor<1x4x27xf32>
%n = "test.op"() : () -> tensor<1x27x15xf32>
%13 = tosa.matmul %m, %n : (tensor<1x4x27xf32>, tensor<1x27x15xf32>) -> tensor<1x4x15xf32>

%14 = "test.op"() : () -> tensor<?x114x114x64xi8>
%15 = tosa.max_pool2d %14 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x114x114x64xi8>) -> tensor<?x56x56x64xi8>
%16 = "test.op"() : () -> tensor<?x25x5x64xi8>
%17 = tosa.avg_pool2d %16 {acc_type = i32, kernel = array<i64: 25, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 25, 5>} : (tensor<?x25x5x64xi8>) -> tensor<?x1x1x64xi8>

%cond = "test.op"() : () -> tensor<i1>
%18 = tosa.cond_if %cond : tensor<i1> -> tensor<12x13xf32> {
  tosa.yield %f : tensor<12x13xf32>
} else {
  tosa.yield %f : tensor<12x13xf32>
}
%19 = tosa.cond_if %cond (%20 = %f) : tensor<i1> (tensor<12x13xf32>) -> tensor<12x13xf32> {
^0(%20 : tensor<12x13xf32>):
  tosa.yield %20 : tensor<12x13xf32>
} else {
^1(%21 : tensor<12x13xf32>):
  tosa.yield %21 : tensor<12x13xf32>
}

// CHECK: builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> tensor<12x34xi32>
// CHECK-NEXT:   %1 = "test.op"() : () -> tensor<1x1xi32>
// CHECK-NEXT:   %2 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %3 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %4 = tosa.add %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %5 = tosa.add %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %6 = tosa.sub %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %7 = tosa.sub %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %f = "test.op"() : () -> tensor<12x13xf32>
// CHECK-NEXT:   %s = "test.op"() : () -> tensor<1xi8>
// CHECK-NEXT:   %8 = tosa.mul %f, %f, %s : (tensor<12x13xf32>, tensor<12x13xf32>, tensor<1xi8>) -> tensor<12x13xf32> 
// CHECK-NEXT:   %9 = tosa.mul %f, %f : (tensor<12x13xf32>, tensor<12x13xf32>) -> tensor<12x13xf32> 
// CHECK-NEXT:   %10 = tosa.mul %0, %1, %s : (tensor<12x34xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<12x34xi32>
// CHECK-NEXT:   %11 = tosa.sin %f : (tensor<12x13xf32>) -> tensor<12x13xf32>
// CHECK-NEXT:   %12 = tosa.cos %f : (tensor<12x13xf32>) -> tensor<12x13xf32>
// CHECK-NEXT:   %m = "test.op"() : () -> tensor<1x4x27xf32>
// CHECK-NEXT:   %n = "test.op"() : () -> tensor<1x27x15xf32>
// CHECK-NEXT:   %13 = tosa.matmul %m, %n : (tensor<1x4x27xf32>, tensor<1x27x15xf32>) -> tensor<1x4x15xf32>
// CHECK-NEXT:   %14 = "test.op"() : () -> tensor<?x114x114x64xi8>
// CHECK-NEXT:   %15 = tosa.max_pool2d %14 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x114x114x64xi8>) -> tensor<?x56x56x64xi8>
// CHECK-NEXT:   %16 = "test.op"() : () -> tensor<?x25x5x64xi8>
// CHECK-NEXT:   %17 = tosa.avg_pool2d %16 {acc_type = i32, kernel = array<i64: 25, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 25, 5>} : (tensor<?x25x5x64xi8>) -> tensor<?x1x1x64xi8>
// CHECK-NEXT:   %cond = "test.op"() : () -> tensor<i1>
// CHECK-NEXT:   %18 = tosa.cond_if %cond : tensor<i1> -> tensor<12x13xf32> {
// CHECK-NEXT:     tosa.yield %f : tensor<12x13xf32>
// CHECK-NEXT:   } else {
// CHECK-NEXT:     tosa.yield %f : tensor<12x13xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %19 = tosa.cond_if %cond (%20 = %f) : tensor<i1> (tensor<12x13xf32>) -> tensor<12x13xf32> {
// CHECK-NEXT:   ^0(%20 : tensor<12x13xf32>):
// CHECK-NEXT:     tosa.yield %20 : tensor<12x13xf32>
// CHECK-NEXT:   } else {
// CHECK-NEXT:   ^1(%21 : tensor<12x13xf32>):
// CHECK-NEXT:     tosa.yield %21 : tensor<12x13xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
