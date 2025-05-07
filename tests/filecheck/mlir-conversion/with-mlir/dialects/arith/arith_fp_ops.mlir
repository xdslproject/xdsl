// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
  %1 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %3 = "arith.negf"(%0) : (f32) -> f32
  %4 = "arith.negf"(%1) : (f64) -> f64
}) : ()->()

// CHECK:        "arith.negf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK:        "arith.negf"(%1) <{fastmath = #arith.fastmath<none>}> : (f64) -> f64
