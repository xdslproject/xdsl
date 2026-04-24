// RUN: mlir-opt %s --allow-unregistered-dialect | xdsl-opt | filecheck %s

// CHECK:        %0 = "test.op"() : () -> i32
// CHECK-NEXT:   %1 = arith.bitcast %0 : i32 to i32
// CHECK-NEXT:   %2 = arith.bitcast %0 : i32 to f32
// CHECK-NEXT:   %3 = arith.bitcast %2 : f32 to i32

// CHECK-NEXT:   %4 = "test.op"() : () -> i64
// CHECK-NEXT:   %5 = arith.bitcast %4 : i64 to f64
// CHECK-NEXT:   %6 = arith.bitcast %5 : f64 to i64

// CHECK-NEXT:   %7 = "test.op"() : () -> i16
// CHECK-NEXT:   %8 = arith.bitcast %7 : i16 to bf16
// CHECK-NEXT:   %9 = arith.bitcast %8 : bf16 to i16
// CHECK-NEXT:   %10 = arith.bitcast %8 : bf16 to f16
// CHECK-NEXT:   %11 = arith.bitcast %10 : f16 to bf16

// CHECK-NEXT:   %12 = "test.op"() : () -> i80
// CHECK-NEXT:   %13 = arith.bitcast %12 : i80 to f80
// CHECK-NEXT:   %14 = arith.bitcast %13 : f80 to i80

// CHECK-NEXT:   %15 = "test.op"() : () -> i128
// CHECK-NEXT:   %16 = arith.bitcast %15 : i128 to f128
// CHECK-NEXT:   %17 = arith.bitcast %16 : f128 to i128

%0 = "test.op"() : () -> i32
%1 = "arith.bitcast"(%0) : (i32) -> i32
%2 = "arith.bitcast"(%0) : (i32) -> f32
%3 = "arith.bitcast"(%2) : (f32) -> i32

%4 = "test.op"() : () -> i64
%5 = "arith.bitcast"(%4) : (i64) -> f64
%6 = "arith.bitcast"(%5) : (f64) -> i64

%7 = "test.op"() : () -> i16
%8 = "arith.bitcast"(%7) : (i16) -> bf16
%9 = "arith.bitcast"(%8) : (bf16) -> i16
%10 = "arith.bitcast"(%8) : (bf16) -> f16
%11 = "arith.bitcast"(%10) : (f16) -> bf16

%12 = "test.op"() : () -> i80
%13 = "arith.bitcast"(%12) : (i80) -> f80
%14 = "arith.bitcast"(%13) : (f80) -> i80

%15 = "test.op"() : () -> i128
%16 = "arith.bitcast"(%15) : (i128) -> f128
%17 = "arith.bitcast"(%16) : (f128) -> i128
