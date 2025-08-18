// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%0 = "test.op"() : () -> tensor<12x34xi32>
%1 = "test.op"() : () -> tensor<1x1xi32>
%2 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%3 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%4 = tosa.add %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%5 = tosa.add %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
%6 = tosa.sub %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
%7 = tosa.sub %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
%8 = "test.op"() : () -> tensor<12x13xf32>
%9 = "test.op"() : () -> tensor<1xi8>
%10 = tosa.mul %8, %8, %9 : (tensor<12x13xf32>, tensor<12x13xf32>, tensor<1xi8>) -> tensor<12x13xf32> 
%11 = tosa.mul %8, %8 : (tensor<12x13xf32>, tensor<12x13xf32>) -> tensor<12x13xf32> 
%12 = tosa.mul %0, %1, %9 : (tensor<12x34xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<12x34xi32>
%13 = tosa.sin %8 : (tensor<12x13xf32>) -> tensor<12x13xf32>
%14 = tosa.cos %8 : (tensor<12x13xf32>) -> tensor<12x13xf32>
%15 = "test.op"() : () -> tensor<1x4x27xf32>
%16 = "test.op"() : () -> tensor<1x27x15xf32>
%17 = tosa.matmul %15, %16 : (tensor<1x4x27xf32>, tensor<1x27x15xf32>) -> tensor<1x4x15xf32>
%18 = "test.op"() : () -> tensor<?x114x114x64xi8>
%19 = tosa.max_pool2d %18 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x114x114x64xi8>) -> tensor<?x56x56x64xi8>
%20 = "test.op"() : () -> tensor<?x25x5x64xi8>
%21 = tosa.avg_pool2d %20 {acc_type = i32, kernel = array<i64: 25, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 25, 5>} : (tensor<?x25x5x64xi8>) -> tensor<?x1x1x64xi8>


// CHECK: module {
// CHECK-NEXT: %0 = "test.op"() : () -> tensor<12x34xi32>
// CHECK-NEXT: %1 = "test.op"() : () -> tensor<1x1xi32>
// CHECK-NEXT: %2 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: %3 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: %4 = tosa.add %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: %5 = tosa.add %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: %6 = tosa.sub %0, %0 : (tensor<12x34xi32>, tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: %7 = tosa.sub %0, %1 : (tensor<12x34xi32>, tensor<1x1xi32>) -> tensor<12x34xi32>
// CHECK-NEXT: %8 = "test.op"() : () -> tensor<12x13xf32>
// CHECK-NEXT: %9 = "test.op"() : () -> tensor<1xi8>
// CHECK-NEXT: %10 = tosa.mul %8, %8, %9 : (tensor<12x13xf32>, tensor<12x13xf32>, tensor<1xi8>) -> tensor<12x13xf32> 
// CHECK-NEXT: %11 = tosa.mul %8, %8 : (tensor<12x13xf32>, tensor<12x13xf32>) -> tensor<12x13xf32> 
// CHECK-NEXT: %12 = tosa.mul %0, %1, %9 : (tensor<12x34xi32>, tensor<1x1xi32>, tensor<1xi8>) -> tensor<12x34xi32>
// CHECK-NEXT: %13 = tosa.sin %8 : (tensor<12x13xf32>) -> tensor<12x13xf32>
// CHECK-NEXT: %14 = tosa.cos %8 : (tensor<12x13xf32>) -> tensor<12x13xf32>
// CHECK-NEXT: %15 = "test.op"() : () -> tensor<1x4x27xf32>
// CHECK-NEXT: %16 = "test.op"() : () -> tensor<1x27x15xf32>
// CHECK-NEXT: %17 = tosa.matmul %15, %16 : (tensor<1x4x27xf32>, tensor<1x27x15xf32>) -> tensor<1x4x15xf32>
// CHECK-NEXT: %18 = "test.op"() : () -> tensor<?x114x114x64xi8>
// CHECK-NEXT: %19 = tosa.max_pool2d %18 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x114x114x64xi8>) -> tensor<?x56x56x64xi8>
// CHECK-NEXT: %20 = "test.op"() : () -> tensor<?x25x5x64xi8>
// CHECK-NEXT: %21 = tosa.avg_pool2d %20 {acc_type = i32, kernel = array<i64: 25, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 25, 5>} : (tensor<?x25x5x64xi8>) -> tensor<?x1x1x64xi8>
// CHECK-NEXT: }
