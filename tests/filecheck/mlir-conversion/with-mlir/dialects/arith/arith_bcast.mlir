// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.bitcast"(%0) : (i32) -> i32
  %2 = "arith.bitcast"(%0) : (i32) -> f32
  %3 = "arith.bitcast"(%2) : (f32) -> i32

  %4 = "arith.constant"() {"value" = 1 : i64} : () -> i64
  %5 = "arith.bitcast"(%4) : (i64) -> f64
  %6 = "arith.bitcast"(%5) : (f64) -> i64
}) : ()->()

// CHECK:        %c1_i32 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:   %0 = "arith.bitcast"(%c1_i32) : (i32) -> i32
// CHECK-NEXT:   %1 = "arith.bitcast"(%c1_i32) : (i32) -> f32
// CHECK-NEXT:   %2 = "arith.bitcast"(%1) : (f32) -> i32
// CHECK-NEXT:   %c1_i64 = "arith.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %3 = "arith.bitcast"(%c1_i64) : (i64) -> f64
// CHECK-NEXT:   %4 = "arith.bitcast"(%3) : (f64) -> i64
