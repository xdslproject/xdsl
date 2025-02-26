// RUN: mlir-opt %s --allow-unregistered-dialect | xdsl-opt | filecheck %s

// CHECK:        %0 = "test.op"() : () -> i32
// CHECK-NEXT:   %1 = arith.bitcast %0 : i32 to i32
// CHECK-NEXT:   %2 = arith.bitcast %0 : i32 to f32
// CHECK-NEXT:   %3 = arith.bitcast %2 : f32 to i32

// CHECK-NEXT:   %4 = "test.op"() : () -> i64
// CHECK-NEXT:   %5 = arith.bitcast %4 : i64 to f64
// CHECK-NEXT:   %6 = arith.bitcast %5 : f64 to i64

%0 = "test.op"() : () -> i32
%1 = "arith.bitcast"(%0) : (i32) -> i32
%2 = "arith.bitcast"(%0) : (i32) -> f32
%3 = "arith.bitcast"(%2) : (f32) -> i32

%4 = "test.op"() : () -> i64
%5 = "arith.bitcast"(%4) : (i64) -> f64
%6 = "arith.bitcast"(%5) : (f64) -> i64
