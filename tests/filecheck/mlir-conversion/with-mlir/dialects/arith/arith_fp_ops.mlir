// RUN: mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic --allow-unregistered-dialect | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
  %1 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %2 = "test.op"() : () -> bf16
  %3 = "test.op"() : () -> f80
  %4 = "test.op"() : () -> f128
  %5 = "arith.negf"(%0) : (f32) -> f32
  %6 = "arith.negf"(%1) : (f64) -> f64
  %7 = "arith.negf"(%2) : (bf16) -> bf16
  %8 = "arith.negf"(%3) : (f80) -> f80
  %9 = "arith.negf"(%4) : (f128) -> f128
}) : ()->()

// CHECK:        "arith.negf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK:        "arith.negf"(%1) <{fastmath = #arith.fastmath<none>}> : (f64) -> f64
// CHECK:        "arith.negf"(%2) <{fastmath = #arith.fastmath<none>}> : (bf16) -> bf16
// CHECK:        "arith.negf"(%3) <{fastmath = #arith.fastmath<none>}> : (f80) -> f80
// CHECK:        "arith.negf"(%4) <{fastmath = #arith.fastmath<none>}> : (f128) -> f128
