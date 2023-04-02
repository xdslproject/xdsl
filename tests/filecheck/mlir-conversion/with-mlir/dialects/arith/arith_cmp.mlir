// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({
  %1 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %2 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %3 = "arith.cmpf"(%1, %2) {"predicate" = 2 : i64} : (f64, f64) -> i1
  %4 = "arith.select"(%3, %1, %2) : (i1, f64, f64) -> f64
}) : ()->()

// CHECK:        "arith.cmpf"(%0, %1) {"predicate" = 2 : i64} : (f64, f64) -> i1
// CHECK:        "arith.select"(%2, %0, %1) : (i1, f64, f64) -> f64