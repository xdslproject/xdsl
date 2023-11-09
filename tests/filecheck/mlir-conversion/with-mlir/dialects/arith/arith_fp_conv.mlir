// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1.0 : f16} : () -> f16
  %1 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
  %2 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %3 = "arith.extf"(%0) : (f16) -> f32
  %4 = "arith.extf"(%0) : (f16) -> f64
  %5 = "arith.extf"(%1) : (f32) -> f64
  %6 = "arith.truncf"(%1) : (f32) -> f16
  %7 = "arith.truncf"(%2) : (f64) -> f32
  %8 = "arith.truncf"(%2) : (f64) -> f16
}) : ()->()

// CHECK:        "arith.extf"(%0) : (f16) -> f32
// CHECK:        "arith.extf"(%0) : (f16) -> f64
// CHECK:        "arith.extf"(%1) : (f32) -> f64
// CHECK:        "arith.truncf"(%1) : (f32) -> f16
// CHECK:        "arith.truncf"(%2) : (f64) -> f32
// CHECK:        "arith.truncf"(%2) : (f64) -> f16
