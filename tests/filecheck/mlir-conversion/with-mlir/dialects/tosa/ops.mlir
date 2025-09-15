// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%0 = "test.op"() : () -> tensor<12x34xi32>
%1 = tosa.clamp %0 {min_fp = 0.0 : f32, max_fp = 1.0: f32, min_int = 0 : i64, max_int = 1 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%2 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
%3 = "test.op"() : () -> tensor<?x114x114x64xi8>
%4 = tosa.max_pool2d %3 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x114x114x64xi8>) -> tensor<?x56x56x64xi8>
%5 = "test.op"() : () -> tensor<?x25x5x64xi8>
%6 = tosa.avg_pool2d %5 {acc_type = i32, kernel = array<i64: 25, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 25, 5>} : (tensor<?x25x5x64xi8>) -> tensor<?x1x1x64xi8>
%7 = tosa.concat %6, %6 {axis = 1 : i32} : (tensor<?x1x1x64xi8>, tensor<?x1x1x64xi8>) -> tensor<?x2x1x64xi8>


// CHECK: module {
// CHECK-NEXT:   %0 = "test.op"() : () -> tensor<12x34xi32>
// CHECK-NEXT:   %1 = tosa.clamp %0 {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %2 = tosa.rescale %0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<12x34xi32>) -> tensor<12x34xi32>
// CHECK-NEXT:   %3 = "test.op"() : () -> tensor<?x114x114x64xi8>
// CHECK-NEXT:   %4 = tosa.max_pool2d %3 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x114x114x64xi8>) -> tensor<?x56x56x64xi8>
// CHECK-NEXT:   %5 = "test.op"() : () -> tensor<?x25x5x64xi8>
// CHECK-NEXT:   %6 = tosa.avg_pool2d %5 {acc_type = i32, kernel = array<i64: 25, 5>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 25, 5>} : (tensor<?x25x5x64xi8>) -> tensor<?x1x1x64xi8>
// CHECK-NEXT:   %7 = tosa.concat %6, %6 {axis = 1 : i32} : (tensor<?x1x1x64xi8>, tensor<?x1x1x64xi8>) -> tensor<?x2x1x64xi8>
// CHECK-NEXT: }
