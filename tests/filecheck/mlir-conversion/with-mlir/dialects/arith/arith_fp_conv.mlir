// RUN: mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic --allow-unregistered-dialect | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1.0 : f16} : () -> f16
  %1 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
  %2 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %bf = "test.op"() : () -> bf16
  %f80v = "test.op"() : () -> f80
  %f128v = "test.op"() : () -> f128
  %3 = "arith.extf"(%0) : (f16) -> f32
  %4 = "arith.extf"(%0) : (f16) -> f64
  %5 = "arith.extf"(%1) : (f32) -> f64
  %6 = "arith.truncf"(%1) : (f32) -> f16
  %7 = "arith.truncf"(%2) : (f64) -> f32
  %8 = "arith.truncf"(%2) : (f64) -> f16
  %9 = "arith.extf"(%bf) : (bf16) -> f32
  %10 = "arith.extf"(%bf) : (bf16) -> f64
  %11 = "arith.extf"(%1) : (f32) -> f80
  %12 = "arith.extf"(%2) : (f64) -> f128
  %13 = "arith.truncf"(%1) : (f32) -> bf16
  %14 = "arith.truncf"(%2) : (f64) -> bf16
  %15 = "arith.truncf"(%f80v) : (f80) -> f32
  %16 = "arith.truncf"(%f128v) : (f128) -> f64
}) : ()->()

// CHECK:        "arith.extf"(%0) : (f16) -> f32
// CHECK:        "arith.extf"(%0) : (f16) -> f64
// CHECK:        "arith.extf"(%1) : (f32) -> f64
// CHECK:        "arith.truncf"(%1) : (f32) -> f16
// CHECK:        "arith.truncf"(%2) : (f64) -> f32
// CHECK:        "arith.truncf"(%2) : (f64) -> f16
// CHECK:        "arith.extf"(%{{.*}}) : (bf16) -> f32
// CHECK:        "arith.extf"(%{{.*}}) : (bf16) -> f64
// CHECK:        "arith.extf"(%1) : (f32) -> f80
// CHECK:        "arith.extf"(%2) : (f64) -> f128
// CHECK:        "arith.truncf"(%1) : (f32) -> bf16
// CHECK:        "arith.truncf"(%2) : (f64) -> bf16
// CHECK:        "arith.truncf"(%{{.*}}) : (f80) -> f32
// CHECK:        "arith.truncf"(%{{.*}}) : (f128) -> f64
